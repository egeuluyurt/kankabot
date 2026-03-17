#!/bin/bash
# ============================================================
# Kanka Sinyal Botu v4.0 — Hetzner Ubuntu 24.04 Kurulum Scripti
# Kullanım: sudo bash setup.sh
# ============================================================
set -e

echo "=================================================="
echo "  Kanka Sinyal Botu v4.0 — VPS Kurulum Başlıyor"
echo "=================================================="

# ── 1. Sistem güncellemesi ─────────────────────────────────
echo "[1/9] Sistem güncelleniyor..."
apt update && apt upgrade -y
apt install -y curl wget git ufw fail2ban python3.11 python3.11-venv python3-pip

# ── 2. Güvenli kullanıcı oluştur ──────────────────────────
echo "[2/9] kankabot kullanıcısı oluşturuluyor..."
if ! id "kankabot" &>/dev/null; then
    useradd -m -s /bin/bash kankabot
    usermod -aG sudo kankabot
    echo "kankabot kullanıcısı oluşturuldu."
else
    echo "kankabot kullanıcısı zaten var, atlanıyor."
fi

# ── 3. SSH sıkılaştırma ────────────────────────────────────
echo "[3/9] SSH sıkılaştırılıyor..."
SSHD_CONFIG="/etc/ssh/sshd_config"
cp "$SSHD_CONFIG" "${SSHD_CONFIG}.backup.$(date +%Y%m%d)"

sed -i 's/^#\?PasswordAuthentication.*/PasswordAuthentication no/' "$SSHD_CONFIG"
sed -i 's/^#\?PermitRootLogin.*/PermitRootLogin no/' "$SSHD_CONFIG"
sed -i 's/^#\?Port.*/Port 2222/' "$SSHD_CONFIG"

# Ayar yoksa ekle
grep -q "^PasswordAuthentication" "$SSHD_CONFIG" || echo "PasswordAuthentication no" >> "$SSHD_CONFIG"
grep -q "^PermitRootLogin" "$SSHD_CONFIG" || echo "PermitRootLogin no" >> "$SSHD_CONFIG"
grep -q "^Port" "$SSHD_CONFIG" || echo "Port 2222" >> "$SSHD_CONFIG"

systemctl restart sshd
echo "SSH portu 2222'ye alındı, root girişi ve şifre girişi devre dışı."

# ── 4. UFW güvenlik duvarı ─────────────────────────────────
echo "[4/9] UFW güvenlik duvarı yapılandırılıyor..."
ufw default deny incoming
ufw default allow outgoing
ufw allow 2222/tcp comment 'SSH'
ufw --force enable
echo "UFW aktif. Sadece port 2222 (SSH) açık."

# ── 5. Fail2Ban kurulumu ───────────────────────────────────
echo "[5/9] Fail2Ban yapılandırılıyor..."
cat > /etc/fail2ban/jail.local << 'EOF'
[DEFAULT]
bantime  = 86400
findtime = 600
maxretry = 5

[sshd]
enabled  = true
port     = 2222
filter   = sshd
logpath  = %(sshd_log)s
maxretry = 5
bantime  = 86400
EOF

systemctl enable fail2ban
systemctl restart fail2ban
echo "Fail2Ban aktif (5 hatalı giriş → 24 saat ban)."

# ── 6. Bot dizini ve sanal ortam ───────────────────────────
echo "[6/9] Python sanal ortamı kuruluyor..."
mkdir -p /opt/kanka-bot
python3.11 -m venv /opt/kanka-bot/venv

# requirements.txt varsa kur
if [ -f "$(dirname "$0")/requirements.txt" ]; then
    cp "$(dirname "$0")/requirements.txt" /opt/kanka-bot/
    /opt/kanka-bot/venv/bin/pip install --upgrade pip
    /opt/kanka-bot/venv/bin/pip install -r /opt/kanka-bot/requirements.txt
    echo "Python bağımlılıkları kuruldu."
else
    echo "UYARI: requirements.txt bulunamadı. Manuel kurulum gerekli."
    echo "  /opt/kanka-bot/venv/bin/pip install -r /opt/kanka-bot/requirements.txt"
fi

# ── 7. Bot dosyalarını kopyala ─────────────────────────────
echo "[7/9] Bot dosyaları kopyalanıyor..."
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
for f in bot.py config.env; do
    if [ -f "$SCRIPT_DIR/$f" ]; then
        cp "$SCRIPT_DIR/$f" /opt/kanka-bot/
        echo "  $f kopyalandı."
    fi
done

# ── 8. Dosya izinleri ──────────────────────────────────────
echo "[8/9] Dosya izinleri ayarlanıyor..."
chmod 600 /opt/kanka-bot/config.env 2>/dev/null || true
chown -R kankabot:kankabot /opt/kanka-bot
echo "config.env: sadece sahip okuyabilir (600)."

# ── 9. Systemd servisi ────────────────────────────────────
echo "[9/9] Systemd servisi yükleniyor..."
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
if [ -f "$SCRIPT_DIR/kanka-bot.service" ]; then
    cp "$SCRIPT_DIR/kanka-bot.service" /etc/systemd/system/
else
    cat > /etc/systemd/system/kanka-bot.service << 'EOF'
[Unit]
Description=Kanka Sinyal Botu v4.0
After=network.target

[Service]
Type=simple
User=kankabot
WorkingDirectory=/opt/kanka-bot
ExecStart=/opt/kanka-bot/venv/bin/python /opt/kanka-bot/bot.py
Restart=always
RestartSec=30
Environment=PYTHONUNBUFFERED=1
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ReadWritePaths=/opt/kanka-bot

[Install]
WantedBy=multi-user.target
EOF
fi

systemctl daemon-reload
systemctl enable kanka-bot
echo "Servis yüklendi ve boot'ta otomatik başlayacak şekilde etkinleştirildi."
echo "(Henüz BAŞLATILMADI — önce config.env doldurulmalı)"

# ── Kurulum tamamlandı ────────────────────────────────────
echo ""
echo "=================================================="
echo "  KURULUM TAMAMLANDI"
echo "=================================================="
echo ""
echo "SONRAKİ ADIMLAR:"
echo ""
echo "1. API anahtarlarını girin:"
echo "   nano /opt/kanka-bot/config.env"
echo ""
echo "2. Botu başlatın:"
echo "   sudo systemctl start kanka-bot"
echo ""
echo "3. Durumu kontrol edin:"
echo "   sudo systemctl status kanka-bot"
echo "   sudo journalctl -u kanka-bot -f"
echo ""
echo "4. Logları izleyin:"
echo "   tail -f /opt/kanka-bot/bot.log"
echo ""
echo "UYARI: Bu sunucuya artık yalnızca port 2222 üzerinden bağlanabilirsiniz!"
echo "  ssh -p 2222 kankabot@<sunucu-ip>"
echo ""
