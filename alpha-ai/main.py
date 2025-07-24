import argparse
import os

def analyze_code(file_path):
    """
    یک فایل کد را تحلیل کرده و اطلاعات اولیه‌ای در مورد آن ارائه می‌دهد.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        num_lines = len(lines)
        num_comments = 0
        for line in lines:
            if line.strip().startswith('#'):  # برای پایتون
                num_comments += 1

        print(f"تحلیل فایل: {file_path}")
        print(f"تعداد کل خطوط: {num_lines}")
        print(f"تعداد کامنت‌ها: {num_comments}")

    except FileNotFoundError:
        print(f"خطا: فایل {file_path} پیدا نشد.")
    except Exception as e:
        print(f"خطایی در هنگام تحلیل فایل رخ داد: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="تحلیلگر اولیه کد برای پروژه آلفا")
    parser.add_argument("file", help="مسیر فایل کدی که باید تحلیل شود.")
    args = parser.parse_args()

    analyze_code(args.file)
