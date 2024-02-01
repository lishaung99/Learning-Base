#需要web框架
#pip install Flask
from flask import Flask,render_template
from random import randint

app = Flask(__name__)#创建一个应用
'''
hero = ['黑暗', '猪八戒',' 沙和尚', '蜘蛛侠', '奥特曼']
@app.route('/index')
def index():
    return render_template('index.html', hero  = hero)
@app.route('/choujiang')
def choujiang():
    num = randint(0, len(hero)-1 )
    return render_template('index.html', hero = hero, h = hero[num])
'''


app.run(debug=True)#启动应用