import smtplib
import os
from dotenv import load_dotenv

load_dotenv()

def send_email_alert(to_email: str, subject: str, body: str) -> None:
    """
    Send an email alert using Gmail SMTP.
    
    Args:
        to_email (str): The recipient's email address
        subject (str): The email subject line
        body (str): The email body content
        
    Returns:
        None
        
    Note:
        Requires environment variables:
        - EMAIL_APP: Gmail address to send from
        - EMAIL_APP_PASSWORD: Gmail app password
        
    Example:
        >>> send_email_alert("user@example.com", "Stock Alert", "AAPL is above $150")
    """
    sender_email = os.getenv("EMAIL_APP")
    sender_pass = os.getenv("EMAIL_APP_PASSWORD")
    
    if not sender_pass:
        print("Error: App password not found in environment.")
        return
    
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, sender_pass)
            server.sendmail(sender_email, to_email, f"Subject: {subject}\n\n{body}")
            print("Email sent successfully.")
    except Exception as e:
        print("Failed to send email:", e)


#send_email_alert("himunagapure114@gmail.com", "Testing notify.py", "Test successful")
