from flask import Flask, render_template, request
import numpy as np

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


from sklearn.metrics import mean_absolute_error,mean_squared_error, max_error,r2_score
from table import *

app = Flask(__name__,static_folder="static")
app.config["SQLALCHEMY_DATABASE_URI"] = "postgresql://postgres:1234567890@localhost:5434/ĐỒ ÁN 1"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db.init_app(app)  

#-------------------------------------------------------------------------TRANG CHỦ------------------------------------------------------------------------------

@app.route('/trangchu1',methods=['GET'])
def trangchu1():
    return render_template('trangchu.html')

@app.route('/add2', methods=['GET','POST'])
def add2():
    return render_template('')

#---------------------------------------------------------------------------QUẢN LÝ---------------------------------------------------------------------------------
@app.route('/quanly')
def quanly():
    return render_template('manager.html')

@app.route('/manager', methods=['GET','POST'])
def manager():
    return render_template('manager.html')

#------------------------------------------------------------------------MÔ HÌNH DỰ ĐOÁN GDP TỔNG-------------------------------------------------------------------

data=pd.read_csv('data/GDP_total.csv')
data1=data.drop(['năm'], axis=1)
# print(data1)
x_vip=data1['tổng dân'].values
y_vip=data1['tổng GDP'].values

x_vip = x_vip.reshape(-1,1)
y_vip = y_vip.reshape(-1,1)
# print(x_vip)
# print(y_vip)
x_train, x_test, y_train, y_test = train_test_split(x_vip,y_vip,train_size=0.8,shuffle=True, random_state=50)
data_total=LinearRegression()
data_total.fit(x_train,y_train)


y_pred = data_total.predict(x_test)
y_pred=np.round(y_pred,0).astype(int)
 
mae = mean_absolute_error(y_pred,y_test)
mse = mean_squared_error(y_pred,y_test)
rmse= (np.sqrt(mae))
max= max_error(y_pred,y_test)



@app.route('/totals')
def totals():
    user = Count.query.filter_by(Name='mô hình dự đoán GDP tổng dân').first()
    
    if user:
        user.number_count +=1
    else:
        user = Count(Name='mô hình dự đoán GDP tổng dân', number_count=1)
        db.session.add(user)
    db.session.commit()
    return render_template('gdptotal.html')
 



@app.route('/predict3', methods=['GET', 'POST'])
def predict3():
    if request.method == 'POST':
        năm = request.form['năm']
        total_population = request.form['total_population']
        
        
        prediction = data_total.predict([[int(total_population)]])
        predicted_data = int(prediction.tolist()[0][0])
        
        totall=total(năm=int(năm), total_population=int(total_population), total_gdp=predicted_data)
        

        db.session.add(totall)
        db.session.commit()
        
        return render_template('gdptotal.html', prediction=predicted_data, mae=mae,mse=mse,rmse=rmse,total_population=total_population, năm=năm)
    


#_______________________TÌM KIẾM_______________________________________   
    

@app.route('/search1', methods=['GET','POST']) 
def search1():
    totall = None
    if request.method == 'POST':
        năm = request.form['năm']
        totall = total.query.filter_by(năm=năm).first()
    return render_template('search/search.html',totall=totall )   
    
   


#________________________________XÓA__________________________________

@app.route('/delete1')
def delete1():
    return render_template("delete/delete1.html")  

@app.route('/delete11', methods=['GET','POST'])
def delete11():
    if request.method == 'POST':
        năm = request.form['năm']
        totall = total.query.filter_by(năm=năm).first()
        db.session.delete(totall)
        db.session.commit()           
    return render_template("delete/delete1.html", NOTE='XÓA THÀNH CÔNG ')    
    
#________________________________UPDATE__________________________________
@app.route('/update1')
def update1():
    return render_template("update/update1.html")     


@app.route('/update11', methods=['GET', 'POST'])
def update11():
    if request.method == 'POST':
        năm = request.form['năm']
        total_population = request.form['total_population']
        total_gdp = request.form['total_gdp']

        totall = total.query.get(năm)
         
        totall.total_population = total_population
        totall.total_gdp = total_gdp 
     
    
        db.session.commit()
    return render_template("update/update1.html", NOTE='UPDATE THÀNH CÔNG')
#--------------------------------------------------------------------MÔ HÌNH DỰ ĐOÁN GDP VÙNG------------------------------------------------------------------------------

df = pd.read_csv('data/region gdp+population2.csv')
df=df.drop(['năm'], axis=1)



X = df.iloc[:, 0:6].values
y = df.iloc[:, 6:].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,shuffle=True, random_state=50)

#huân luyện mô hình
model = LinearRegression()
model.fit(X_train, y_train)

# đặt biến y để tính sai số mae
y_pred = model.predict(X_test)
y_pred=np.round(y_pred,0).astype(int)
 
mae = mean_absolute_error(y_pred,y_test)
mse = mean_squared_error(y_pred,y_test)
rmse= (np.sqrt(mae))

@app.route('/region')
def vip():
    user = Count.query.filter_by(Name='mô hình dự đoán GDP vùng').first()
    
    if user:
        user.number_count +=1
    else:
        user = Count(Name='mô hình dự đoán GDP vùng', number_count=1)
        db.session.add(user)
    db.session.commit()
    return render_template('gdvregion.html')

@app.route('/predicts', methods=['GET', 'POST'])
def predicts():
    if request.method == 'POST':
        năm = request.form['năm']
        sông_hồng = request.form['SONGHONG']
        trung_du = request.form['TRUNGDU']
        bắc_trung_bộ = request.form['BACTRUNGBO']
        cửu_long = request.form['CUULONG']
        đông_nam_bộ = request.form['DONGNAMBO']
        tây_nguyên = request.form['TAYNGUYEN']
        #tạo 1 array để hiển thi các giá trị nhập vào
        save = (int(sông_hồng),int(trung_du),int(bắc_trung_bộ),int(cửu_long),int(đông_nam_bộ),int(tây_nguyên))

        prediction2 = model.predict([[int(sông_hồng),int(trung_du),int(bắc_trung_bộ),int(cửu_long),int(đông_nam_bộ),int(tây_nguyên)]])
        predicted_data2 = int(prediction2.tolist()[0][0])
        
     


        region = regions(năm=năm,sông_hồng=int(sông_hồng),  trung_du=int(trung_du) , bắc_trung_bộ=int(bắc_trung_bộ),  cửu_long=int(cửu_long),   đông_nam_bộ=int(đông_nam_bộ), tây_nguyên=int(tây_nguyên),gdp_vung=predicted_data2)
        db.session.add(region)
        db.session.commit()
        
        return render_template('gdvregion.html', prediction=predicted_data2,ĐỘC_LẬP='{}'.format('  ; '.join(map(str, save))), mae=mae,mse=mse,rmse=rmse, năm=năm)
    
 
#_______________________TÌM KIẾM_______________________________________
 
@app.route('/search2', methods=['GET','POST']) 
def search2():   
    region = None     
    if request.method == 'POST':
        năm = request.form['năm']
        region = regions.query.filter_by(năm=năm).first()
    return render_template('search/search2.html',  region=region)     




#_________________________XÓA_________________________________________

@app.route('/delete2')
def delete2():
    return render_template("delete/delete2.html")  

@app.route('/delete22', methods=['GET','POST'])
def delete22():
    if request.method == 'POST':
        năm = request.form['năm']
        region = regions.query.filter_by(năm=năm).first()
        db.session.delete(region)
        db.session.commit()           
    return render_template("delete/delete2.html", NOTE='XÓA THÀNH CÔNG')     

#________________________________UPDATE__________________________________


@app.route('/update2')
def update2():
    return render_template("update/update2.html")     


@app.route('/update22', methods=['GET', 'POST'])
def update22():
    if request.method == 'POST':
        năm = request.form['năm']
        sông_hồng = request.form['SONGHONG']
        trung_du = request.form['TRUNGDU']
        bắc_trung_bộ = request.form['BACTRUNGBO']
        cửu_long = request.form['CUULONG']
        đông_nam_bộ = request.form['DONGNAMBO']
        tây_nguyên = request.form['TAYNGUYEN']
        gdp_vung = request.form['gdp_vung']
                                     
        region = regions.query.get(năm)
         
        region.sông_hồng = sông_hồng
        region.trung_du = trung_du
        region.bắc_trung_bộ = bắc_trung_bộ
        region.cửu_long = cửu_long 
        region.đông_nam_bộ = đông_nam_bộ
        region.tây_nguyên = tây_nguyên 
        region.gdp_vung = gdp_vung 
        db.session.commit()
    return render_template("update/update2.html", NOTE='UPDATE THÀNH CÔNG')

#--------------------------------------------------------------------MÔ HÌNH DỰ ĐOÁN GDP TUỔI-----------------------------------------------------------------------------------
dataage=pd.read_csv('data/GDP age-work.csv')
dataage2=dataage.drop(['TỔNG GDP 15-24','TỔNG GDP 25-49','TỔNG GDP 50+'], axis=1)
dataage2


x_age=dataage2[['LAO ĐỘNG 15-24','LAO ĐỘNG 25-49','LAO ĐỘNG 50+']].values
y_age=dataage2[['TotalGDP-lđ']].values


x_train, x_test, y_train, y_test = train_test_split(x_age,y_age,train_size=0.8,shuffle=True, random_state=50)
data_age=LinearRegression()
data_age.fit(x_train,y_train)


y_pred = data_age.predict(x_test)
y_pred=np.round(y_pred,0).astype(int)
 
mae = mean_absolute_error(y_pred,y_test)
mse = mean_squared_error(y_pred,y_test)
rmse= (np.sqrt(mae))
max= max_error(y_pred,y_test)

@app.route('/agess')
def agess():
    user = Count.query.filter_by(Name='mô hình dự đoán GDP tuổi').first()
    if user:
        user.number_count +=1
    else:
        user = Count(Name='mô hình dự đoán GDP tuổi', number_count=1)
        db.session.add(user)
    db.session.commit()
    return render_template('gdpages.html')
   
   

@app.route('/predict5', methods=['GET', 'POST'])
def predict5():
    if request.method == 'POST':
        năm = request.form['năm']
        t15_24 = request.form['15_24']
        t25_49 = request.form['25_49']
        t50 = request.form['50']
        
        save=(int(t15_24), int(t25_49), int(t50))
        prediction5 = data_age.predict([[int(t15_24), int(t25_49), int(t50)]])
        predicted_data5 = int(prediction5.tolist()[0][0])
        
        ages=age(năm=int(năm), t15_24=int(t15_24), t25_49=int(t25_49), t50=int(t50), gdp_age=predicted_data5)
        db.session.add(ages)
        db.session.commit()

        return render_template('gdpages.html', data=predicted_data5, INPUT = '{}'.format('  ; '.join(map(str, save))),mae=mae,mse=mse, rmse=rmse, max=max,năm=năm)
    


#_______________________TÌM KIẾM_______________________________________

@app.route('/search3', methods=['GET','POST']) 
def search3():   
    ages = None     
    if request.method == 'POST':
        năm = request.form['năm']
        ages = age.query.filter_by(năm=năm).first()
    return render_template('search/search3.html',  ages=ages)     



#_________________________XÓA_________________________________________

@app.route('/delete3')
def delete3():
    return render_template("delete/delete3.html")  

@app.route('/delete33', methods=['GET','POST'])
def delete33():
    if request.method == 'POST':
        năm = request.form['năm']
        ages=age.query.filter_by(năm=năm).first()
        db.session.delete(ages)
        db.session.commit()           
    return render_template("delete/delete3.html", NOTE='XÓA THÀNH CÔNG')    



#________________________________UPDATE__________________________________


@app.route('/update3')
def update3():
    return render_template("update/update3.html")     


@app.route('/update33', methods=['GET', 'POST'])
def update33():
    if request.method == 'POST':
        năm = request.form['năm']
        t15_24 = request.form['15_24']
        t25_49 = request.form['25_49']
        t50 = request.form['50']
        gdp_age = request.form['gdp_age']  

        ages=age.query.get(năm)
        ages.t15_24= t15_24
        ages.t25_49= t25_49
        ages.t50= t50 
        ages.gdp_age= gdp_age 
        db.session.commit()
    return render_template("update/update3.html", NOTE='UPDATE THÀNH CÔNG')

    
#------------------------------------------------------------------MÔ HÌNH DỰ ĐOÁN GDP THÀNH THỊ VÀ NÔNG THÔN-------------------------------------------------------------------------

datacity=pd.read_csv('data/GDP_city_and_country.csv')
# print(datacity)
x_city=datacity[['dân số (nông thôn)','dân số (Thành Thị)']].values
y_city=datacity[['total gdp']].values
# print(x_city)
# print(y_city)

x_train, x_test, y_train, y_test = train_test_split(x_city,y_city,train_size=0.8,shuffle=True, random_state=50)
data_city=LinearRegression()
data_city.fit(x_train,y_train)


y_pred2 = data_city.predict(x_test)
y_pred2=np.round(y_pred2,0).astype(int)
 
mae1 = mean_absolute_error(y_pred2,y_test)
mse1 = mean_squared_error(y_pred2,y_test)
rmse1= (np.sqrt(mae1))
max= max_error(y_pred2,y_test)
        

@app.route('/city')
def city():
    user = Count.query.filter_by(Name = 'Mô hình dự đoán GDP thành thị-nông thôn').first()
    if user:
        user.number_count +=1
    else:
        user = Count(Name = 'Mô hình dự đoán GDP thành thị-nông thôn', number_count=1)
        db.session.add(user)
    db.session.commit()
    return render_template('gdpcity.html')

@app.route('/predict4', methods=['GET', 'POST'])
def predict4():
    if request.method == 'POST':
        năm = request.form['năm']
        nt = request.form['nt']
        tt = request.form['tt']
        
        save=(int(nt), int(tt))
        prediction4 = data_city.predict([[int(nt), int(tt)]])
        predicted_data4 = int(prediction4.tolist()[0][0])
        
        citys=city_contry(năm=int(năm), nt=int(nt),tt=int(tt), gdp_city=predicted_data4)
        db.session.add(citys)
        db.session.commit()

        return render_template('gdpcity.html', data=predicted_data4,ĐỘC_LẬP='{}'.format(' ; '.join(map(str, save))), mae1=mae1,mse1=mse1, rmse1=rmse1, max=max, năm=năm)
    


#_______________________TÌM KIẾM_______________________________________
    


@app.route('/search4', methods=['GET','POST'])
def search4():
    citys = None
    if request.method == 'POST':
        năm = request.form['năm']
        citys = city_contry.query.filter_by(năm=năm).first()
    return render_template('search/search4.html', citys=citys)


#_________________________XÓA_________________________________________


@app.route('/delete4')
def delete4():
    return render_template("delete/delete4.html")  

@app.route('/delete44', methods=['GET','POST'])
def delete44():
    if request.method == 'POST':
        năm = request.form['năm']
        citys=city_contry.query.filter_by(năm=năm).first()
        db.session.delete(citys)
        db.session.commit()           
    return render_template("delete/delete4.html", NOTE='XÓA THÀNH CÔNG')   

#________________________________UPDATE__________________________________


@app.route('/update4')
def update4():
    return render_template("update/update4.html")     


@app.route('/update44', methods=['GET', 'POST'])
def update44():
    if request.method == 'POST':
        năm = request.form['năm']
        nt = request.form['nt']
        tt = request.form['tt']
        gdp_city = request.form['gdp_city']

        citys = city_contry.query.get(năm)
        citys.nt= nt
        citys.tt= tt
        citys.gdp_city= gdp_city 
         
        db.session.commit()
    return render_template("update/update4.html", NOTE='UPDATE THÀNH CÔNG')




#------------------------------------------------------XEM-----------------------------------------------------------------------
@app.route('/view', methods=['GET', 'POST'])
def view():
    ages =age.query.all()
    region = regions.query.all()
    citys = city_contry.query.all()
    totall = total.query.all()
    return render_template('view/view.html',data1=data1, ages=ages, totall=totall,region=region,citys=citys)

 

#-----------------------------------------------------XEM SỐ LẦN ĐẾM------------------------------------------------------------------
@app.route('/count')
def count():
       #ĐẾM MÔ HÌNH THÀNH THỊ VÀ NÔNG THÔN
    city = Count.query.filter_by(Name = 'Mô hình dự đoán GDP thành thị-nông thôn').first()
    if city:  
        city_count = city.number_count
    else:
       city_count = 0
    #ĐẾM MÔ HÌNH TỔNG DÂN
    total = Count.query.filter_by(Name='mô hình dự đoán GDP tổng dân').first()
    if total:
        total_count = total.number_count
    else:
        total_count=0
   #ĐẾM MÔ HÌNH VÙNG
    region = Count.query.filter_by(Name='mô hình dự đoán GDP vùng').first()
    if region:
        region_count=region.number_count
    else:
        region_count=0
#ĐẾM MÔ HÌNH TUỔI
    age = Count.query.filter_by(Name='mô hình dự đoán GDP tuổi').first()
    if age:
        age_count=age.number_count
    else:
        age_count=0

    return render_template('view/view2.html', city_count=city_count,total_count=total_count,region_count=region_count,age_count=age_count) 

if __name__ == '__main__':
    app.run(debug=True)