from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle
from model_building import Combine_Transformer  # 导入自定义的类

# 自定义反序列化方法
def load_model(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

# 加载训练好的模型
model = load_model("model.pkl")

app = Flask(__name__)

@app.route('/', methods=["GET", "POST"])
def index():
    pred = ""
    if request.method == "POST":
        # 获取表单输入数据
        price = request.form.get("price", 0)
        promotion = request.form.get("promotion", 0)
        unit_count_comb = request.form.get("Unit_Count_comb", 0)
        brand = request.form.get("Brand", "")
        manufacturer = request.form.get("Manufacturer", "")
        flavor = request.form.get("Flavor", "")
        primary_supplement = request.form.get("Primary_Supplement", "")
        item_form = request.form.get("Item_Form", "")
        age_range = request.form.get("Age_Range", "")
        diet_type = request.form.get("Diet_Type", "")
        directions = request.form.get("Directions", "")
        description = request.form.get("Description", "")
        benefit_comb = request.form.get("Benefit", "")
        ingredient_comb = request.form.get("Ingredient", "")
        
        # 将所有输入数据合并为一个DataFrame
        X = pd.DataFrame({
            'price': [price],
            'promotion': [promotion],
            'Unit Count_comb': [unit_count_comb],
            'Brand': [brand],
            'Manufacturer': [manufacturer],
            'Flavor_comb': [flavor],
            'Primary Supplement Type_comb': [primary_supplement],
            'Item_Form_updt': [item_form],
            'Age Range_comb': [age_range],
            'Diet Type_comb': [diet_type],
            'Directions_updt': [directions],
            'Description_comb': [description],
            'Benefit_comb': [benefit_comb],
            'Ingredient_comb': [ingredient_comb]
        })

        # 打印数据框以供调试
        print("Input DataFrame:", X)

        # 使用模型预测结果，直接返回流行与否的判断
        pred = model.predict(X)[0]  # 假设模型返回的是 0（不流行）或 1（流行）

    return render_template("index.html", pred=pred)

if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5000)

