//! The cap's Wi-Fi client, through wpa_supplicant's control socket (`/var/run/wpa_supplicant/wlan0`, a UNIX datagram socket;
//! the cap's config has `ctrl_interface` and `update_config=1`, so SAVE_CONFIG writes `/userdata/wpa_supplicant.conf`).
//! The vendor's own MQTT path is not used: it can delete network 0 and add nothing, and it logs the password.

use std::collections::HashSet;
use std::fs;
use std::os::unix::net::UnixDatagram;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;

use serde_json::{Value, json};

pub struct Wpa {
    ctrl: PathBuf,
}

/// A saved network: its wpa_supplicant id and SSID.
struct Saved {
    id: u32,
    ssid: String,
}

impl Wpa {
    pub fn new(ctrl: impl Into<PathBuf>) -> Self {
        Self { ctrl: ctrl.into() }
    }

    /// Saves `ssid` (or updates it if saved) with `password` (empty = an open network) and enables it; the saved networks
    /// keep their entries. The password is never echoed back.
    pub fn join(&self, ssid: &str, password: &str) -> Result<Value, String> {
        // WPA2 passphrases are 8-63 printable ASCII characters; wpa_supplicant prints a bad one in its error line.
        if !password.is_empty() && !(8..=63).contains(&password.len()) || !password.bytes().all(|b| (0x20..0x7f).contains(&b)) {
            return Err("the password must be 8-63 printable characters".into());
        }
        if ssid.is_empty() || ssid.len() > 32 {
            return Err("the network name must be 1-32 bytes".into());
        }
        let (id, added) = match self.saved()?.into_iter().find(|n| n.ssid == ssid) {
            Some(saved) => (saved.id, false),
            None => (self.request("ADD_NETWORK")?.parse().map_err(|_| "wpa_supplicant did not add a network".to_string())?, true),
        };
        let configure = || -> Result<(), String> {
            self.ok(&format!("SET_NETWORK {id} ssid {}", hex(ssid)))?;
            if password.is_empty() {
                self.ok(&format!("SET_NETWORK {id} key_mgmt NONE"))?;
            } else {
                self.ok(&format!("SET_NETWORK {id} key_mgmt WPA-PSK"))?;
                self.ok(&format!("SET_NETWORK {id} psk \"{password}\""))?;
            }
            self.ok(&format!("ENABLE_NETWORK {id}"))
        };
        if let Err(error) = configure() {
            // A half-made entry must not reach the file at the next SAVE_CONFIG.
            if added && let Err(undo) = self.ok(&format!("REMOVE_NETWORK {id}")) {
                return Err(format!("{error}; {undo}"));
            }
            return Err(error);
        }
        self.ok("SAVE_CONFIG")?;
        Ok(json!({"id": id, "ssid": ssid}))
    }

    /// What the Wi-Fi card shows: the link (STATUS), the saved networks, and the networks the last scan found (strongest
    /// first, one row per name, hidden names left out). Never a password: wpa_supplicant does not hand them out.
    pub fn state(&self) -> Result<Value, String> {
        let status: Vec<(String, String)> =
            self.request("STATUS")?.lines().filter_map(|l| l.split_once('=')).map(|(k, v)| (k.to_string(), v.to_string())).collect();
        let field = |key: &str| status.iter().find(|(k, _)| k == key).map(|(_, v)| v.clone());
        let saved = self.saved()?;
        let mut seen: Vec<(String, i32, u32, bool)> = Vec::new();
        for line in self.request("SCAN_RESULTS")?.lines().skip(1) {
            let fields: Vec<&str> = line.splitn(5, '\t').collect();
            let [_, freq, signal, flags, ssid] = fields[..] else { continue };
            let secured = ["WPA", "RSN", "WEP", "SAE"].iter().any(|k| flags.contains(k));
            seen.push((unescape(ssid), signal.parse().unwrap_or(-100), freq.parse().unwrap_or(0), secured));
        }
        // Strongest first; the sort is stable, so the first row of a name is its strongest.
        seen.sort_by_key(|(_, signal, ..)| -signal);
        let mut names = HashSet::new();
        seen.retain(|(ssid, ..)| !ssid.is_empty() && names.insert(ssid.clone()));
        Ok(json!({
            "ssid": field("ssid").map(|s| unescape(&s)),
            "ip": field("ip_address"),
            "freq": field("freq").and_then(|f| f.parse::<u32>().ok()),
            "saved": saved.iter().map(|n| json!({"id": n.id, "ssid": n.ssid})).collect::<Vec<_>>(),
            "seen": seen.iter().map(|(ssid, signal, freq, secured)| json!({
                "ssid": ssid, "signal": signal, "freq": freq, "secured": secured, "saved": saved.iter().any(|n| n.ssid == *ssid),
            })).collect::<Vec<_>>(),
        }))
    }

    /// Asks for a new scan; the results arrive in [`Wpa::state`] a few seconds later (FAIL-BUSY: a scan is already going).
    pub fn scan(&self) -> Result<Value, String> {
        match self.request("SCAN")?.as_str() {
            "OK" | "FAIL-BUSY" => Ok(json!({"scanning": true})),
            reply => Err(format!("wpa_supplicant refused SCAN ({reply})")),
        }
    }

    /// Removes saved network `id`.
    pub fn forget(&self, id: u32) -> Result<Value, String> {
        self.ok(&format!("REMOVE_NETWORK {id}"))?;
        self.ok("SAVE_CONFIG")?;
        Ok(json!({"id": id}))
    }

    /// The saved networks (LIST_NETWORKS: a header line, then `id \t ssid \t bssid \t flags`).
    fn saved(&self) -> Result<Vec<Saved>, String> {
        Ok(self
            .request("LIST_NETWORKS")?
            .lines()
            .skip(1)
            .filter_map(|line| {
                let mut fields = line.split('\t');
                Some(Saved { id: fields.next()?.parse().ok()?, ssid: unescape(fields.next()?) })
            })
            .collect())
    }

    /// A command that must answer OK.
    fn ok(&self, command: &str) -> Result<(), String> {
        match self.request(command)?.as_str() {
            "OK" => Ok(()),
            reply => Err(format!("wpa_supplicant refused {} ({reply})", verb(command))),
        }
    }

    /// One command and its reply. Errors name the command's first word only: a SET_NETWORK carries the password.
    fn request(&self, command: &str) -> Result<String, String> {
        static NEXT: AtomicU32 = AtomicU32::new(0);
        let local = std::env::temp_dir().join(format!("robocap-panel-wpa-{}-{}", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
        let _ = fs::remove_file(&local);
        let exchange = || -> std::io::Result<String> {
            let socket = UnixDatagram::bind(&local)?;
            socket.set_read_timeout(Some(Duration::from_secs(5)))?;
            socket.connect(&self.ctrl)?;
            socket.send(command.as_bytes())?;
            let mut buffer = vec![0u8; 64 * 1024];
            let n = socket.recv(&mut buffer)?;
            Ok(String::from_utf8_lossy(&buffer[..n]).trim_end().to_string())
        };
        let reply = exchange();
        let _ = fs::remove_file(&local);
        reply.map_err(|error| format!("wpa_supplicant {}: {error}", verb(command)))
    }
}

fn verb(command: &str) -> &str {
    command.split(' ').next().unwrap_or_default()
}

/// An SSID as wpa_supplicant takes it unquoted: hex bytes, so any character is safe.
fn hex(text: &str) -> String {
    text.bytes().map(|b| format!("{b:02x}")).collect()
}

/// wpa_supplicant prints SSIDs with printf-style escapes (`\\`, `\"`, `\xNN`, `\n` ...).
fn unescape(text: &str) -> String {
    let mut bytes = Vec::new();
    let mut chars = text.bytes();
    while let Some(b) = chars.next() {
        if b != b'\\' {
            bytes.push(b);
            continue;
        }
        match chars.next() {
            Some(b'x') => {
                let digits: Vec<u8> = chars.by_ref().take(2).collect();
                bytes.extend(std::str::from_utf8(&digits).ok().and_then(|d| u8::from_str_radix(d, 16).ok()));
            }
            Some(b'n') => bytes.push(b'\n'),
            Some(b'r') => bytes.push(b'\r'),
            Some(b't') => bytes.push(b'\t'),
            Some(b'e') => bytes.push(0x1b),
            Some(other) => bytes.push(other),
            None => {}
        }
    }
    String::from_utf8_lossy(&bytes).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::net::UnixDatagram;
    use std::sync::{Arc, Mutex};

    /// A stand-in for wpa_supplicant 2.10's control socket: the saved networks and every command it got.
    #[derive(Default)]
    struct Fake {
        networks: Vec<(u32, String)>,
        commands: Vec<String>,
        saves: u32,
        /// Answer FAIL to SET_NETWORK psk, as wpa_supplicant does for a passphrase it cannot use.
        refuse_psk: bool,
    }

    /// Serves `fake` on `<dir>/wlan0` with wpa_supplicant's reply formats (ctrl_iface.c) until the test ends.
    fn serve(dir: &std::path::Path, fake: Arc<Mutex<Fake>>) -> PathBuf {
        let path = dir.join("wlan0");
        let socket = UnixDatagram::bind(&path).expect("bind the fake control socket");
        std::thread::spawn(move || {
            let mut buffer = [0u8; 4096];
            while let Ok((n, from)) = socket.recv_from(&mut buffer) {
                let command = String::from_utf8_lossy(&buffer[..n]).to_string();
                let reply = {
                    let mut fake = fake.lock().unwrap();
                    fake.commands.push(command.clone());
                    let words: Vec<&str> = command.splitn(4, ' ').collect();
                    match words.as_slice() {
                        ["LIST_NETWORKS"] => {
                            let mut text = "network id / ssid / bssid / flags\n".to_string();
                            for (id, ssid) in &fake.networks {
                                text += &format!("{id}\t{ssid}\tany\t\n");
                            }
                            text
                        }
                        ["ADD_NETWORK"] => {
                            let id = fake.networks.iter().map(|(id, _)| id + 1).max().unwrap_or(0);
                            fake.networks.push((id, String::new()));
                            format!("{id}\n")
                        }
                        ["SET_NETWORK", id, "ssid", hex] => {
                            let bytes: Vec<u8> = (0..hex.len()).step_by(2).map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap()).collect();
                            let id: u32 = id.parse().unwrap();
                            fake.networks.iter_mut().find(|(n, _)| *n == id).unwrap().1 = String::from_utf8(bytes).unwrap();
                            "OK\n".into()
                        }
                        ["SET_NETWORK", _, "psk", _] if fake.refuse_psk => "FAIL\n".into(),
                        ["REMOVE_NETWORK", id] => {
                            let id: u32 = id.parse().unwrap();
                            fake.networks.retain(|(n, _)| *n != id);
                            "OK\n".into()
                        }
                        ["SET_NETWORK", ..] | ["ENABLE_NETWORK", _] => "OK\n".into(),
                        ["SAVE_CONFIG"] => {
                            fake.saves += 1;
                            "OK\n".into()
                        }
                        _ => "UNKNOWN COMMAND\n".into(),
                    }
                };
                let Some(to) = from.as_pathname() else { continue };
                let _ = socket.send_to(reply.as_bytes(), to);
            }
        });
        path
    }

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("robocap-panel-wifi-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn joining_a_new_network_keeps_the_saved_ones_and_never_echoes_the_password() {
        let fake = Arc::new(Mutex::new(Fake { networks: vec![(0, "PVWifi".into())], ..Fake::default() }));
        let wpa = Wpa::new(serve(&temp_dir("join"), fake.clone()));

        let reply = wpa.join("GL-BE3600-7e5", "correct horse").expect("join");

        let fake = fake.lock().unwrap();
        assert_eq!(fake.networks, vec![(0, "PVWifi".to_string()), (1, "GL-BE3600-7e5".to_string())]);
        assert!(fake.commands.contains(&"SET_NETWORK 1 ssid 474c2d4245333630302d376535".to_string()), "{:?}", fake.commands);
        assert!(fake.commands.contains(&"SET_NETWORK 1 psk \"correct horse\"".to_string()), "{:?}", fake.commands);
        assert!(fake.commands.contains(&"ENABLE_NETWORK 1".to_string()), "{:?}", fake.commands);
        assert_eq!(fake.saves, 1);
        assert!(!reply.to_string().contains("correct horse"), "{reply}");
    }

    #[test]
    fn a_refused_join_leaves_the_saved_networks_as_they_were() {
        let fake = Arc::new(Mutex::new(Fake { networks: vec![(0, "PVWifi".into())], refuse_psk: true, ..Fake::default() }));
        let wpa = Wpa::new(serve(&temp_dir("refused"), fake.clone()));

        let error = wpa.join("GL-BE3600-7e5", "correct horse").expect_err("wpa_supplicant refused the password");

        let fake = fake.lock().unwrap();
        assert_eq!(fake.networks, vec![(0, "PVWifi".to_string())]);
        assert_eq!(fake.saves, 0);
        assert!(!error.contains("correct horse"), "{error}");
    }

    #[test]
    fn forgetting_a_network_removes_only_that_one() {
        let fake = Arc::new(Mutex::new(Fake { networks: vec![(0, "PVWifi".into()), (1, "GL-BE3600-7e5".into())], ..Fake::default() }));
        let wpa = Wpa::new(serve(&temp_dir("forget"), fake.clone()));

        wpa.forget(1).expect("forget");

        let fake = fake.lock().unwrap();
        assert_eq!(fake.networks, vec![(0, "PVWifi".to_string())]);
        assert_eq!(fake.saves, 1);
    }

    #[test]
    fn a_password_wpa_supplicant_cannot_use_is_refused_before_it_is_sent() {
        let fake = Arc::new(Mutex::new(Fake { networks: vec![(0, "PVWifi".into())], ..Fake::default() }));
        let wpa = Wpa::new(serve(&temp_dir("short"), fake.clone()));

        let error = wpa.join("GL-BE3600-7e5", "short").expect_err("7 characters");

        assert!(error.contains("8-63"), "{error}");
        assert!(!error.contains("short"), "{error}");
        assert!(fake.lock().unwrap().commands.is_empty());
    }
}
