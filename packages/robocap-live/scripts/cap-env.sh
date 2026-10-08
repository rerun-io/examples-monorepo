# Sourced by deploy.sh and panel.sh after they parse their arguments: which cap, and how to reach it. The scripts set CAP from
# their required --cap a|b before sourcing; there is no default cap.
# cap_ssh runs a command on the cap through robocap-ssh (found on PATH; it takes a|b and holds the caps' ssh routes and
# addresses). Nothing here names a host or an address: panel.sh takes the cap's address as an argument (--cap-address).
# CAP_HOSTNAME is the hostname the cap must answer with (the scripts refuse any other device).
# Remote layout on either cap: /root/robocap-live/{bin,lib,models,assets,logs,run} (deploy.sh's --root moves it, e.g.
# /root/robocap-live/test for a test deploy that leaves the main binary alone).

CAP_ROOT=/root/robocap-live
case ${CAP:-} in
    a|A)
        CAP=a
        CAP_HOSTNAME=robocap_f403b0
        ;;
    b|B)
        CAP=b
        CAP_HOSTNAME=robocap_fe62fa
        ;;
    '')
        echo "--cap a|b is required (which cap)" >&2
        return 2 2>/dev/null || exit 2
        ;;
    *)
        echo "unknown cap '$CAP' (expected a or b)" >&2
        return 2 2>/dev/null || exit 2
        ;;
esac
cap_ssh() {
    local tool
    tool=$(command -v robocap-ssh) || { echo "robocap-ssh is not on PATH (the helper that reaches a cap: robocap-ssh a|b <command>)" >&2; return 127; }
    "$tool" "$CAP" "$@"
}
export CAP CAP_ROOT CAP_HOSTNAME
export -f cap_ssh
