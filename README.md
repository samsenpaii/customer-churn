# 📊 Internet Service Churn Prediction App

A simple web application built with **Streamlit**, **scikit-learn**, and **Pandas** to predict whether a customer is likely to churn based on their subscription and service data.

---

## 🚀 Features

- Predict customer churn using **Logistic Regression**
- Clean and interactive **Streamlit UI** for input
- Automatically handles **missing values** and **feature scaling**
- Provides **insights and recommendations** for high churn risk users
- Visualizes:
  - **Correlation Matrix**
  - **Feature Importance**

---

## 📁 Dataset

The app uses a CSV dataset named `internet_service_churn.csv` with the following columns:

| Column Name                   | Description                                                  |
|------------------------------|--------------------------------------------------------------|
| `id`                         | Unique customer ID (dropped during preprocessing)            |
| `is_tv_subscriber`           | Whether the customer subscribes to TV services (0 or 1)      |
| `is_movie_package_subscriber`| Movie package subscription (0 or 1)                          |
| `subscription_age`           | Years since subscription                                     |
| `bill_avg`                   | Average of last 3 months' bills                              |
| `remaining_contract`         | Remaining contract period in years                           |
| `service_failure_count`      | Number of service failures in the last 3 months              |
| `download_avg`               | Average data downloaded in GB (last 3 months)                |
| `upload_avg`                 | Average data uploaded in GB (last 3 months)                  |
| `download_over_limit`        | Whether download limit was exceeded (0 or 1)                 |
| `churn`                      | Target variable: 1 = churned, 0 = not churned                |

---

## 🛠 How It Works

1. Loads and preprocesses the data:
   - Drops `id`
   - Fills missing values with median
   - Standardizes features
2. Trains a **Logistic Regression** model
3. Accepts user input via sidebar
4. Predicts **churn probability**
5. If risk is high:
   - Displays **top influencing features**
   - Suggests **actions to reduce churn**
6. Shows **correlation matrix** and **feature importance** visualizations

---

## 🖥 Usage

### 🔧 Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```
### ▶️ Run the App
```bash
streamlit run churn_app.py
```
