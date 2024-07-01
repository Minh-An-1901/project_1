import flask
from table import *

app = flask.Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "postgresql://postgres:1234567890@localhost:5434/ĐỒ ÁN 1"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db.init_app(app)


def main():
    db.create_all()
    db.session.commit()

if __name__ == "__main__":
    with app.app_context():  
        main()