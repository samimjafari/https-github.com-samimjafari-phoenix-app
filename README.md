# Glowing-Guacamole AI Assistant 🥑🚀

یک دستیار هوش مصنوعی حرفه‌ای، آفلاین‌محور با **حافظه معنایی**، **رمزنگاری AES-256-GCM**، **جستجوی وب** و **اتصال GitHub**. طراحی شده برای دسکتاپ، اندروید (Termux) و کالی لینوکس.

---

## 📖 دستورالعمل ماستر نصب و اجرا

### ۱. پیش‌نیازها
- **Android (Termux):**
  ```bash
  pkg update && pkg upgrade
  pkg install python git nodejs npm clang
  ```
- **Kali Linux / Ubuntu / Debian:**
  ```bash
  sudo apt update && sudo apt upgrade
  sudo apt install python3 python3-pip git nodejs npm g++ build-essential
  ```
- **Windows/Linux Desktop:**
  - نصب Python 3.8+
  - نصب Node.js و npm
  - نصب Git

---

### ۲. دریافت سورس کد
```bash
git clone https://github.com/samimjafari/glowing-guacamole.git
cd glowing-guacamole
```

---

### ۳. نصب وابستگی‌ها
این پروژه دو بخش دارد: پایتون (AI Assistant) و جاوااسکریپت (GUI/Build)

- **پایتون:**
  ```bash
  pip install -r requirements.txt
  # یا نصب دستی:
  pip install flet llama-cpp-python sentence-transformers cryptography numpy rich PyGithub beautifulsoup4
  ```

- **Node.js (برای ساخت GUI و APK/EXE):**
  ```bash
  npm install
  ```

---

### ۴. نصب ابزارهای مخصوص هر پلتفرم
- **Termux:**
  ```bash
  chmod +x setup_termux.sh
  ./setup_termux.sh
  ```
- **Kali Linux:**
  ```bash
  chmod +x scripts/install-tools-kali.sh
  ./scripts/install-tools-kali.sh
  ```
- **Linux Desktop:**
  ```bash
  chmod +x scripts/install-tools-linux.sh
  ./scripts/install-tools-linux.sh
  ```

---

### ۵. راه‌اندازی مدل آفلاین
- یک مدل GGUF (مثل TinyLlama) دانلود کنید.
- نام آن را به `your-model.gguf` تغییر دهید.
- در پوشه‌ی اصلی پروژه قرار دهید.

---

### ۶. اجرای برنامه
- **رابط گرافیکی (GUI):**
  ```bash
  python gui_flet.py
  # یا با npm:
  npm start
  ```
- **رابط خط فرمان (CLI):**
  ```bash
  python cli.py
  # یا با npm:
  npm run cli
  ```

---

### ۷. ساخت اپلیکیشن (Build)
- **ساخت EXE (Windows):**
  ```bash
  npm run desktop:build
  ```
- **ساخت APK (Android):**
  ```bash
  npm run mobile:apk
  ```

---

### ۸. نکته امنیتی 🔒
در اولین اجرا، از شما **رمز اصلی (Master Password)** خواسته می‌شود. این رمز برای تولید کلید رمزگذاری **AES-256-GCM** استفاده می‌شود. اگر رمز را فراموش کنید، حافظه و داده‌های شما غیرقابل بازیابی خواهند بود.

---
*Created with ❤️ for the open-source community.*
