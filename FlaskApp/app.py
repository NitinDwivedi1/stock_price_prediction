from flask import Flask, render_template, send_file, request, flash, redirect, url_for
import MySQLdb.cursors
from flask_mysqldb import MySQL
import os
import math
from array import array

from model import predict_price



app = Flask(__name__)
app.secret_key = "stockp"

print("FLASK_ENV:", os.getenv("FLASK_ENV"))


app.config['MYSQL_HOST'] = os.environ.get("MYSQL_HOST")
# app.config['MYSQL_PORT'] = 3306
app.config['MYSQL_USER'] = os.environ.get("MYSQL_USER")
app.config['MYSQL_PASSWORD'] = os.environ.get("MYSQL_PASS")
app.config['MYSQL_DB'] = os.environ.get("MYSQL_DB")

mysql = MySQL(app)

@app.route('/')
def form():
    return render_template('form.html')


@app.route('/data', methods=['POST'])
def predict():

    stock_name = request.form['Name']
    # ep = request.form['Epochs']
    ahead = request.form['Ahead']
    days = 30

    # ep = int(ep)
    ahead = int(ahead)
    days = int(days)

    predicted_price, plot_img = predict_price(stock_name, ahead, days)
    return render_template("plot.html", plot_img=plot_img, predicted_price=predicted_price, ahead=ahead, stock_name=stock_name)


@app.route('/feedback', methods=['POST'])
def feedback():
    print("helloo")
    review_star = request.form['fbstar']
    review_star = float(review_star)
    review_text = request.form['fbtext']
    print(review_star,type(review_star))
    print(review_text)
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute('INSERT INTO stock.feedback VALUES (NULL,%s,%s)', (review_star, review_text))#insert the sql query
    mysql.connection.commit()
    cursor.close()
    msg = 'Thanks for your feedback!'
    flash(msg,"success")
    return redirect(url_for('form'))


if __name__ == '__main__':
    app.run(debug=True)
