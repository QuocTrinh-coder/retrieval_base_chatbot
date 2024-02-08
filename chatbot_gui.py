#Creating GUI with tkinter
import tkinter
from tkinter import *
from chatapp import ChatApp as cA

def send():
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)
    if msg != '':
            ChatLog.config(state=NORMAL)
            ChatLog.insert(END, "You: " + msg + '\n\n')
            ChatLog.config(foreground="#442265", font=("Verdana", 12 ))
            res = cA().chatbot_response(msg)
            ChatLog.insert(END, "Bot: " + res + '\n\n')
            ChatLog.config(state=DISABLED)
            ChatLog.yview(END)


base = Tk()
base.title("ChatBot - SL")
base.geometry("500x600")
base.resizable(width=TRUE, height=TRUE)

#Create Chat window
ChatLog = Text(base, bd=0, bg="white", height="50", width="300", font="Arial",)
ChatLog.config(state=DISABLED)
#Bind scrollbar to Chat window
scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set

#Create Button to send message
SendButton = Button(base, font=("Verdana",12,'bold'), text="Send", width="12", height=5,
                    bd=0, bg="#32de97", activebackground="#3c9d9b",fg='#ffffff',
                    command= send )
#Create the box to enter message
EntryBox = Text(base, bd=0, bg="white",width="50", height="10", font="Arial")
#EntryBox.bind("<Return>", send)

#Place all components on the screen
scrollbar.place(x=520,y=6, height=386)
ChatLog.place(x=6,y=6, height=386, width=500)
EntryBox.place(x=128, y=401, height=90, width=500)
SendButton.place(x=6, y=401, height=90)

base.mainloop()