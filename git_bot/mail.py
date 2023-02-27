import imaplib
import email
from email.header import decode_header
from urllib import parse
import quopri
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import re

import time



class MailGet:
    def __init__(self):
        print(1)
        self.get_mail_count()
        # self.setter()
        # self.start()
        print(2)

    def get_mail_count(self):

        imap, username, mail_pass = self.__go_mail()
        msg = self.__get(imap)
        self.index_test = self.__check(imap)


    def __go_mail(self):
        mail_pass = "pass"
        username = "user_name"
        imap_server = "imap.mail.ru"
        imap = imaplib.IMAP4_SSL(imap_server)
        imap.login(username, mail_pass)
        return imap, username, mail_pass

    def start(self):

        # while True:
            imap, username, mail_pass = self.__go_mail()
            msg = self.__get(imap)
            index = self.__check(imap)
            if self.index_test != index:
                # print('сообщение')
                self.index_test = self.__check(imap)
                self.__send_massage(imap, username, mail_pass)
                self.__check_in = True
                return True

            else:
                # print('пусто')
                self.__check_in = False
                time.sleep(1)
                return False
                # Add a sleep call to prevent the infinite loop from consuming too many resources.
                # self.start()
        # self.setter()

    @property
    def check_in(self):

        return self.__check_in

    def setter(self):
        self.start()
        # self.__check_in = False

    def __send_massage(self,imap, username, mail_pass):
        self.__send_mail(self.__get(imap), username, mail_pass)


    def __get(self, imap):
        res, msg = imap.fetch(imap.select("INBOX")[-1].pop(-1), '(RFC822)')
        return email.message_from_bytes(msg[0][1])

    def __check(self, imap):
        index = int(" ".join(map(str, list.copy(imap.select("INBOX")[-1]))).split("'")[-2])
        return index

    def __normalization(self, string):
        text = re.sub(r'\<[^>]*\>', '', parse.unquote(string, 'utf-8')).replace('\r\n', ' ').replace('  ', ' ')

        return text.partition(';}')[2].replace('  ', '').replace('  ', ' ')

    @property
    def get_text(self):
        return self.__answer

    @property
    def id_task(self):
        return self.task_id

    def __send_mail(self, msg, username, mail_pass):
        for part in msg.walk():
            if part.get_content_maintype() == 'text' and part.get_content_subtype() == 'html':
                decoded_string = quopri.decodestring(part.get_payload()).decode('utf-8')
                task_id = decoded_string.partition('https://itap.planfix.ru/task/')[2].rsplit("'>")[0]
                text = self.__normalization(decoded_string)
                msg = MIMEMultipart()
                msg['From'] = username
                try:

                    subject = decode_header(msg["Subject"])[0][0]
                except:
                    subject = msg["Subject"]
                msg['Subject'] = subject
                body = text + f'https://itap.planfix.ru/task/{task_id}'
                self.task_id = f'https://itap.planfix.ru/task/{task_id}'
                msg.attach(MIMEText(body, 'plain'))
                server = smtplib.SMTP_SSL('smtp.mail.ru', 465)
                server.login(username, mail_pass)
                server.sendmail(username, 'kiselev.ng@pecom.ru', msg.as_string())
                # print(f"{subject}  {body}")
                # self.__answer = f'{subject} {body}'
                self.__answer = f"{subject}  {body}"

#
