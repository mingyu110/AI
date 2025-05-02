import os
import smtplib
import datetime
import time
import schedule
import threading
import re
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from dotenv import load_dotenv
import ssl

# 加载环境变量
load_dotenv()

# 邮件配置
EMAIL_HOST = os.getenv("EMAIL_HOST", "smtp.gmail.com")
EMAIL_PORT = int(os.getenv("EMAIL_PORT", "587"))
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
# 是否使用SSL（端口465通常需要SSL）
USE_SSL = EMAIL_PORT == 465

# 检查邮件配置
if not all([EMAIL_HOST, EMAIL_PORT, EMAIL_USER]):
    print("警告: 邮件配置不完整。请在.env文件中设置EMAIL_HOST, EMAIL_PORT和EMAIL_USER。")

# 全局变量用于跟踪调度线程
scheduler_thread = None
stop_scheduler = False

def send_email(recipient, subject, body, attachment_path=None, password=None, content_type='plain'):
    """
    发送邮件
    
    参数:
    - recipient: 收件人邮箱(字符串或列表，支持多个邮箱地址用逗号、分号或空格分隔)
                 还支持"邮箱:授权码"格式为单个邮箱指定授权码
    - subject: 邮件主题
    - body: 邮件正文
    - attachment_path: 附件路径(可选)
    - password: 邮箱密码(可选，如不提供则使用环境变量中的密码)
    - content_type: 内容类型，'plain'(纯文本)、'html'(HTML格式)或'markdown'(Markdown格式)
    
    返回:
    - 全部成功返回True，全部失败返回False，部分成功返回邮件发送成功率大于50%则True
    """
    # 处理收件人和可能的授权码
    recipients_with_passwords = []
    default_password = password if password else EMAIL_PASSWORD
    
    if isinstance(recipient, list):
        recipients_list = recipient
    else:
        recipients_list = [r.strip() for r in re.split(r'[,;\s]+', recipient) if r.strip()]
    
    # 处理每个收件人
    for recipient_entry in recipients_list:
        # 检查是否包含授权码 (email:password 格式)
        if ':' in recipient_entry:
            email, specific_password = recipient_entry.split(':', 1)
            recipients_with_passwords.append((email.strip(), specific_password.strip()))
        else:
            # 使用默认授权码
            recipients_with_passwords.append((recipient_entry.strip(), default_password))
    
    if not recipients_with_passwords:
        print("未提供有效的收件人邮箱地址")
        return False
    
    # 发送邮件给每个收件人
    success_count = 0
    failure_count = 0
    
    for email, pwd in recipients_with_passwords:
        try:
            # 验证密码
            if not pwd:
                print(f"未提供邮箱密码，请通过参数传入或在环境变量中设置EMAIL_PASSWORD")
                failure_count += 1
                continue
            
            # 确定SMTP服务器设置（根据邮箱类型可能需要不同设置）
            current_host = EMAIL_HOST
            current_port = EMAIL_PORT
            current_ssl = USE_SSL
            
            # 对网易邮箱，自动使用其SMTP服务器
            if "@126.com" in email:
                current_host = "smtp.126.com"
                current_port = 465
                current_ssl = True
            elif "@163.com" in email:
                current_host = "smtp.163.com"
                current_port = 465
                current_ssl = True
            elif "@qq.com" in email:
                current_host = "smtp.qq.com"
                current_port = 465
                current_ssl = True
            
            print(f"邮箱 {email} 使用服务器: {current_host}:{current_port}, SSL: {current_ssl}")
            
            # 创建邮件
            msg = MIMEMultipart('alternative')  # 使用alternative类型支持多种格式
            # 重要：使用当前邮箱作为From地址，而不是固定的EMAIL_USER
            msg['From'] = f"新能源汽车分析 AI 助手 <{email}>"
            msg['To'] = email
            msg['Subject'] = subject
            
            # 处理Markdown内容
            if content_type.lower() == 'markdown':
                # 确保markdown库已导入
                import markdown
                
                # 始终添加原始Markdown作为纯文本版本（保证兼容性）
                plain_part = MIMEText(body, 'plain', 'utf-8')
                msg.attach(plain_part)
                
                # 转换Markdown到HTML
                html_body = markdown.markdown(body, extensions=['tables', 'fenced_code'])
                
                # 添加CSS样式，美化Markdown显示
                styled_html = f"""
                <!DOCTYPE html>
                <html lang="zh-CN">
                <head>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <style>
                        body {{
                            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                            line-height: 1.6;
                            color: #333;
                            max-width: 900px;
                            margin: 0 auto;
                            padding: 20px;
                        }}
                        h1 {{
                            color: #1a5276;
                            border-bottom: 2px solid #3498db;
                            padding-bottom: 10px;
                        }}
                        h2 {{
                            color: #2874a6;
                            border-bottom: 1px solid #3498db;
                            padding-bottom: 5px;
                        }}
                        h3 {{
                            color: #2e86c1;
                        }}
                        a {{
                            color: #3498db;
                            text-decoration: none;
                        }}
                        a:hover {{
                            text-decoration: underline;
                        }}
                        blockquote {{
                            border-left: 4px solid #3498db;
                            padding: 10px 20px;
                            margin: 20px 0;
                            background-color: #f5f9fa;
                        }}
                        code {{
                            background-color: #f5f5f5;
                            padding: 2px 4px;
                            border-radius: 4px;
                            font-family: Menlo, Monaco, "Courier New", monospace;
                        }}
                        pre {{
                            background-color: #f5f5f5;
                            padding: 15px;
                            border-radius: 5px;
                            overflow: auto;
                        }}
                        table {{
                            border-collapse: collapse;
                            width: 100%;
                            margin: 20px 0;
                        }}
                        th, td {{
                            border: 1px solid #ddd;
                            padding: 8px 12px;
                            text-align: left;
                        }}
                        th {{
                            background-color: #f2f2f2;
                        }}
                        tr:nth-child(even) {{
                            background-color: #f9f9f9;
                        }}
                        img {{
                            max-width: 100%;
                            height: auto;
                        }}
                    </style>
                </head>
                <body>
                    {html_body}
                </body>
                </html>
                """
                
                # 使用HTML格式的邮件正文
                html_part = MIMEText(styled_html, 'html', 'utf-8')
                msg.attach(html_part)
                
            else:
                # 添加正文（支持纯文本或HTML格式）
                msg.attach(MIMEText(body, content_type, 'utf-8'))
            
            # 添加附件
            if attachment_path and os.path.exists(attachment_path):
                with open(attachment_path, 'rb') as file:
                    # 如果是Markdown文件，将其作为附件添加
                    if attachment_path.endswith('.md'):
                        # 添加Markdown附件
                        md_attachment = MIMEApplication(
                            file.read(),
                            _subtype='markdown'
                        )
                        md_attachment.add_header('Content-Disposition', 'attachment', filename=os.path.basename(attachment_path))
                        msg.attach(md_attachment)
                    else:
                        # 普通附件
                        attachment = MIMEApplication(file.read(), _subtype='octet-stream')
                        attachment.add_header('Content-Disposition', 'attachment', filename=os.path.basename(attachment_path))
                        msg.attach(attachment)
            
            # 连接SMTP服务器并发送
            # 根据当前邮箱设置选择是否使用SSL
            if current_ssl:
                # 使用SSL连接（适用于网易邮箱等）
                context = ssl.create_default_context()
                with smtplib.SMTP_SSL(current_host, current_port, context=context) as server:
                    # 使用当前邮箱的账户和密码进行认证
                    server.login(email, pwd)
                    server.send_message(msg)
            else:
                # 使用普通TLS连接
                with smtplib.SMTP(current_host, current_port) as server:
                    server.starttls()
                    # 使用当前邮箱的账户和密码进行认证
                    server.login(email, pwd)
                    server.send_message(msg)
            
            success_count += 1
            print(f"邮件已成功发送至 {email}")
            
        except Exception as e:
            failure_count += 1
            print(f"发送邮件到 {email} 失败: {str(e)}")
    
    # 返回发送结果
    if failure_count == 0:
        return True
    elif success_count == 0:
        return False
    else:
        # 部分成功
        print(f"邮件发送结果: {success_count}成功, {failure_count}失败")
        return True if success_count > failure_count else False

def schedule_email(recipient, subject, body, attachment_path=None, send_time=None, password=None, content_type='plain'):
    """
    安排在指定时间发送邮件
    
    参数:
    - recipient: 收件人邮箱(字符串或列表，支持多个邮箱地址用逗号、分号或空格分隔)
                 还支持"邮箱:授权码"格式为单个邮箱指定授权码
    - subject: 邮件主题
    - body: 邮件正文
    - attachment_path: 附件路径(可选)
    - send_time: 发送时间(datetime对象，默认为立即发送)
    - password: 邮箱密码(可选，如不提供则使用环境变量中的密码)
    - content_type: 内容类型，'plain'(纯文本)或'html'(HTML格式)
    
    返回:
    - 成功返回True，失败返回False
    """
    global scheduler_thread, stop_scheduler
    
    # 如果未指定发送时间或发送时间为"now"，立即发送
    if send_time is None or send_time == "now":
        return send_email(recipient, subject, body, attachment_path, password, content_type)
    
    # 计算延迟秒数
    now = datetime.datetime.now()
    if isinstance(send_time, datetime.datetime):
        if send_time < now:
            print("指定的发送时间已过，将立即发送")
            return send_email(recipient, subject, body, attachment_path, password, content_type)
        
        # 保存密码，用于定时任务
        saved_password = password
        saved_content_type = content_type
        
        # 安排发送任务
        def send_task():
            send_email(recipient, subject, body, attachment_path, saved_password, saved_content_type)
        
        # 将发送时间格式化为调度器可以理解的格式
        schedule_time = send_time.strftime("%H:%M:%S")
        schedule_day = send_time.strftime("%Y-%m-%d")
        
        # 如果是今天，使用at来调度
        if send_time.date() == now.date():
            schedule.every().day.at(schedule_time).do(send_task)
            print(f"邮件已安排在今天 {schedule_time} 发送")
        else:
            # 否则，创建一个检查任务
            def check_date_and_send():
                current_date = datetime.datetime.now().strftime("%Y-%m-%d")
                if current_date == schedule_day:
                    send_task()
                    return schedule.CancelJob  # 发送后取消任务
            
            schedule.every().day.at("00:01").do(check_date_and_send)
            print(f"邮件已安排在 {schedule_day} {schedule_time} 发送")
        
        # 启动调度器线程
        if scheduler_thread is None or not scheduler_thread.is_alive():
            stop_scheduler = False
            scheduler_thread = threading.Thread(target=run_scheduler)
            scheduler_thread.daemon = True
            scheduler_thread.start()
        
        return True
    else:
        print("无效的发送时间格式")
        return False

def run_scheduler():
    """运行调度器线程"""
    global stop_scheduler
    
    while not stop_scheduler:
        schedule.run_pending()
        time.sleep(1)

def stop_scheduler_thread():
    """停止调度器线程"""
    global stop_scheduler
    stop_scheduler = True 