from aiogram import Bot, Dispatcher, executor, types

import nltk
import asyncio
from finel_bot.mail import MailGet
from finel_bot.ii_bot import IiPy
import time

nltk.download('punkt')
nltk.download('stopwords')

API_TOKEN = 'token'

bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)


async def check_emails(mails_array: list)->None:
    while True:
        mail_text.start()
        if mail_text.check_in:
            mails_array.append([mail_text.get_text, mail_text.id_task])
        await asyncio.sleep(1)



@dp.message_handler(commands=['start'])
async def send_welcome(message: types.Message) -> None:
    while True:
        if len(mails_text_array) != 0:
            array_task = mails_text_array.pop(-1)
            text = array_task[0]
            task = array_task[-1]
            all_text = ii.start(text)
            answer = f"{all_text[0]} {task}"
            await message.answer(answer)
        await asyncio.sleep(1)


if __name__ == '__main__':
    ii = IiPy()
    temp_massage = []
    mails_text_array = []
    mail_text = MailGet()
    asyncio.ensure_future(check_emails(mails_text_array))
    executor.start_polling(dp, skip_updates=True)
