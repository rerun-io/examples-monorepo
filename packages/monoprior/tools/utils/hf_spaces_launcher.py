import subprocess
from pathlib import Path

PIXI_PATH = Path("/home/user/.pixi/bin/pixi")
BREW_PATH = Path("/home/linuxbrew/.linuxbrew/bin/brew")
LSOF_PATH = Path("/home/linuxbrew/.linuxbrew/Cellar/lsof/4.99.3/bin/lsof")


def check_and_install_pixi() -> None:
    try:
        subprocess.check_call(f"{PIXI_PATH} --version", shell=True)
    except subprocess.CalledProcessError:
        print("pixi not found. Installing pixi...")
        # Install pixi using the provided installation script
        subprocess.check_call(
            "curl -fsSL https://pixi.sh/install.sh | bash", shell=True
        )


def check_and_install_homebrew() -> None:
    try:
        # Check if Homebrew is installed
        subprocess.check_call(f"{BREW_PATH} --version", shell=True)
    except subprocess.CalledProcessError:
        # If Homebrew is not found, install it
        print("Homebrew not found. Installing Homebrew...")
        subprocess.check_call(
            '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"',
            shell=True,
        )


def install_package(package_name) -> None:
    try:
        # Install the specified package using Homebrew
        subprocess.check_call(f"{BREW_PATH} install {package_name}", shell=True)
        print(f"{package_name} installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install {package_name}. Error: {e}")


def run_command(command: str) -> None:
    try:
        subprocess.check_call(command, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"run command {command}. Error: {e}")


if __name__ == "__main__":
    check_and_install_homebrew()
    install_package(package_name="lsof")
    check_and_install_pixi()
    run_command(command=f"{LSOF_PATH} -t -i:7860 | xargs -r kill")
    run_command(command=f"{PIXI_PATH} run -e spaces app")
