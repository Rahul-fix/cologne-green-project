# Google Drive Sync Setup (via rclone)

Since you have a 2TB Google One plan, you can use Google Drive to store your processed data instead of GCS. We will use `rclone`, a powerful command-line tool for this.

## 1. Install rclone (on VM)
```bash
sudo -v ; curl https://rclone.org/install.sh | sudo bash
```

## 2. Configure rclone (Local Machine)
Since your VM is headless (no browser), you need to generate the config on your **local machine** first and then copy it to the VM.

### On your Local Mac:
1.  Install rclone: `brew install rclone`
2.  Run config: `rclone config`
3.  Follow these steps:
    *   **n** (New remote)
    *   name: **gdrive**
    *   Storage: **drive** (Google Drive)
    *   client_id: Leave blank (or use your own if you have one)
    *   client_secret: Leave blank
    *   scope: **1** (Full access)
    *   service_account_file: Leave blank
    *   Edit advanced config? **n**
    *   Use web browser? **y**
    *   (Your browser will open. Log in to your Google Account and allow access.)
    *   Configure this as a Shared Drive? **n** (unless you use one)
    *   **y** (Yes, this is OK)
    *   **q** (Quit)

4.  View the config:
    ```bash
    rclone config show
    ```
    Copy the entire output (it looks like `[gdrive] type = drive ...`).

## 3. Configure rclone (on VM)
1.  On your VM, run:
    ```bash
    mkdir -p ~/.config/rclone
    nano ~/.config/rclone/rclone.conf
    ```
2.  Paste the config you copied from your local machine.
3.  Save and exit (`Ctrl+O`, `Enter`, `Ctrl+X`).

## 4. Verify Access
Run this on the VM to list your Drive files:
```bash
rclone ls gdrive:
```

## 5. Sync Data
We have provided a script `scripts/sync_to_drive.py` to automate the upload.

```bash
# Sync to a folder named 'cologne-green-project' in your Drive
uv run python scripts/sync_to_drive.py gdrive:cologne-green-project
```
