
from flask_sqlalchemy import SQLAlchemy        # KHỞI TẠO BẢNG(4 BẢNG VÀ 5 TRƯỜNG)      
db = SQLAlchemy()           
  
class Count(db.Model):
    ID = db.Column(db.Integer, primary_key=True)
    Name = db.Column(db.String, nullable=False)
    number_count = db.Column(db.Integer, nullable=False) 
class regions(db.Model):
    năm = db.Column(db.Integer, primary_key=True)
    sông_hồng = db.Column(db.Integer)
    trung_du = db.Column(db.Integer)
    bắc_trung_bộ = db.Column(db.Integer)
    cửu_long = db.Column(db.Integer)
    đông_nam_bộ = db.Column(db.Integer)
    tây_nguyên = db.Column(db.Integer)
    gdp_vung = db.Column(db.Integer)
class age(db.Model):
    năm = db.Column(db.Integer, primary_key=True)
    t15_24 = db.Column(db.Integer)
    t25_49 = db.Column(db.Integer)
    t50 = db.Column(db.Integer)
    gdp_age = db.Column(db.Integer)
class total(db.Model):
    năm = db.Column(db.Integer, primary_key=True)
    total_population = db.Column(db.Integer, nullable=False)
    total_gdp = db.Column(db.Integer, nullable=False)

class city_contry(db.Model):
    năm = db.Column(db.Integer, primary_key=True)
    nt = db.Column(db.Integer)
    tt = db.Column(db.Integer)
    gdp_city = db.Column(db.Integer)



   