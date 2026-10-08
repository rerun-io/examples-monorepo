//! The Rerun viewer chosen by name: the Mac's Bonjour name (e.g. `MacBook-Pro-4`), saved on the cap in `run/viewer-name`.
//! At each look-up the cap asks every network it is on (its Wi-Fi client and its hotspot) "who is <name>.local?" with a
//! one-shot mDNS query (RFC 6762 6.7: from an ephemeral port, so the Mac answers by unicast), then checks that the viewer port
//! answers. The cap has no mDNS resolver of its own (no avahi, no nss-mdns), so robocap-live gets the address, not the name.

use std::net::{Ipv4Addr, SocketAddr, TcpStream, UdpSocket};
use std::path::Path;
use std::time::{Duration, Instant};

use serde_json::{Value, json};

const VIEWER_PORT: u16 = 9876;
const MDNS: (Ipv4Addr, u16) = (Ipv4Addr::new(224, 0, 0, 251), 5353);

/// The saved viewer name, or the default.
fn name(root: &Path) -> String {
    crate::read_trim(root.join("run/viewer-name")).filter(|n| valid(n)).unwrap_or_else(|| "MacBook-Pro-4".into())
}

pub fn save_name(root: &Path, name: &str) -> Result<Value, String> {
    let name = name.trim().trim_end_matches(".local");
    if !valid(name) {
        return Err("a viewer name is 1-63 letters, digits or '-' (the Mac's Local hostname)".into());
    }
    std::fs::write(root.join("run/viewer-name"), name).map_err(|e| format!("cannot save the viewer name: {e}"))?;
    Ok(json!({"name": name}))
}

fn valid(name: &str) -> bool {
    (1..=63).contains(&name.len()) && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '-')
}

/// Where the named viewer is now: its address on a network the cap shares with it, and whether the viewer port answers.
pub fn find(root: &Path) -> Value {
    let name = name(root);
    let Some(address) = resolve(&name) else {
        return json!({"name": name, "found": false, "error": format!("{name} not found on the cap's networks (is the Mac awake and on the same Wi-Fi?)")});
    };
    let running = TcpStream::connect_timeout(&SocketAddr::from((address, VIEWER_PORT)), Duration::from_millis(800)).is_ok();
    json!({
        "name": name, "found": true, "address": address.to_string(), "running": running,
        "url": format!("rerun+http://{address}:{VIEWER_PORT}/proxy"),
    })
}

/// Asks all of the cap's IPv4 networks at once for `<name>.local` and waits up to 1.5 s (a Mac on Wi-Fi took 291 ms to
/// answer at home); the first answer wins.
fn resolve(name: &str) -> Option<Ipv4Addr> {
    let query = query(name);
    let sockets: Vec<UdpSocket> = addresses()
        .into_iter()
        .filter_map(|local| {
            // Bound to the interface's own address, the multicast goes out of that interface.
            let socket = UdpSocket::bind((local, 0)).ok()?;
            let _ = socket.set_multicast_ttl_v4(255);
            socket.send_to(&query, MDNS).ok()?;
            socket.set_nonblocking(true).ok()?;
            Some(socket)
        })
        .collect();
    if sockets.is_empty() {
        return None;
    }
    let deadline = Instant::now() + Duration::from_millis(1500);
    let mut buffer = [0u8; 1500];
    while Instant::now() < deadline {
        for socket in &sockets {
            while let Ok(n) = socket.recv(&mut buffer) {
                if let Some(address) = answer(&buffer[..n], name) {
                    return Some(address);
                }
            }
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    None
}

/// The cap's IPv4 addresses except loopback, from `ip -4 -o addr` ("3: wlan0    inet 192.168.1.218/24 brd ...").
fn addresses() -> Vec<Ipv4Addr> {
    crate::stdout_of("ip", &["-4", "-o", "addr"])
        .lines()
        .filter_map(|line| line.split_whitespace().skip_while(|w| *w != "inet").nth(1)?.split('/').next()?.parse::<Ipv4Addr>().ok())
        .filter(|address| !address.is_loopback())
        .collect()
}

/// One question: `<name>.local`, type A, class IN with the unicast-response bit.
fn query(name: &str) -> Vec<u8> {
    let mut packet = vec![0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0];
    for label in [name, "local"] {
        packet.push(label.len() as u8);
        packet.extend_from_slice(label.as_bytes());
    }
    packet.extend_from_slice(&[0, 0, 1, 0x80, 1]);
    packet
}

/// The A record for `<name>.local` in a DNS message, if any (answers and additional records; names may be compressed).
fn answer(packet: &[u8], name: &str) -> Option<Ipv4Addr> {
    let count = |at: usize| Some(u16::from_be_bytes([*packet.get(at)?, *packet.get(at + 1)?]) as usize);
    let (questions, records) = (count(4)?, count(6)? + count(8)? + count(10)?);
    let mut at = 12;
    for _ in 0..questions {
        at = read_name(packet, at)?.1 + 4;
    }
    let wanted = format!("{name}.local");
    for _ in 0..records {
        let (owner, end) = read_name(packet, at)?;
        let kind = count(end)?;
        let length = count(end + 8)?;
        let data = packet.get(end + 10..end + 10 + length)?;
        if kind == 1 && length == 4 && owner.eq_ignore_ascii_case(&wanted) {
            return Some(Ipv4Addr::new(data[0], data[1], data[2], data[3]));
        }
        at = end + 10 + length;
    }
    None
}

/// A DNS name at `at` (following compression pointers) and the offset just after it in the record.
fn read_name(packet: &[u8], mut at: usize) -> Option<(String, usize)> {
    let mut labels = Vec::new();
    let mut end = None;
    for _ in 0..64 {
        let length = *packet.get(at)? as usize;
        match length {
            0 => return Some((labels.join("."), end.unwrap_or(at + 1))),
            l if l & 0xc0 == 0xc0 => {
                end.get_or_insert(at + 2);
                at = ((l & 0x3f) << 8) | *packet.get(at + 1)? as usize;
            }
            l => {
                labels.push(String::from_utf8_lossy(packet.get(at + 1..at + 1 + l)?).to_string());
                at += 1 + l;
            }
        }
    }
    None
}
