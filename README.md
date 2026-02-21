# LoanGuard 🛡️
**Smart Loan Risk Assessment & Bank Matching Engine**

🚀 **[Live Demo: Try LoanGuard Here](https://loanguard-65dz.onrender.com/)**
## 📌 Project Overview
Getting a loan can be a highly opaque process. Applicants often don't know their true default risk or which banks are most likely to offer them the best rates based on their specific financial profile. 

**LoanGuard** bridges this gap. It is a full-stack financial technology application that uses Machine Learning to instantly evaluate an applicant's financial health, calculate their default risk, and cross-reference current market data to recommend the most optimal banking partners in India.

---

## ✨ Key Features
* **Instant Risk Analysis:** Evaluates 9 different financial data points to generate an accurate default risk percentage.
* **Smart Bank Matching:** Doesn't just give a score—it actively matches users with real-world Indian banks offering the best interest rates for their profile.
* **Dynamic Interest Calculation:** Estimates a personalized interest rate based on the user's Credit Score, Debt-to-Income (DTI) ratio, and Employment History.
* **Enterprise-Grade UI:** A clean, responsive, and intuitive frontend built for a seamless user experience.

---

## ⚙️ How It Works

### 1. The Inputs (What the user provides)
The user fills out a secure form providing basic financial parameters:
* **Demographics:** Age, Employment Status, Years Employed.
* **Financials:** Annual Income, Existing Debt, Credit Score.
* **Loan Details:** Desired Loan Amount, Loan Type, and Tenure (1, 3, or 5 years).

### 2. The Engine (Why it works)
Once the data is submitted, it is processed through our FastAPI backend:
* **Data Scaling:** The inputs are standardized using a pre-trained `StandardScaler` to match the format of our training data.
* **Machine Learning Model:** A pre-trained **Random Forest Classifier** analyzes the scaled data. It looks at complex relationships (like `loan_to_income` and `income_stability`) to predict the probability of default.
* **Suitability Algorithm:** The system queries our live database of Indian Banks. It calculates a "Match Score" by weighing the bank's base interest rate against the applicant's calculated risk factor.

### 3. The Outputs (What the user gets)
* **Risk Verdict:** A clear categorization (Low, Moderate, or High Risk) with a visual progress bar.
* **Estimated Interest Rate:** A realistic expectation of the interest rate they qualify for.
* **Top Bank Recommendations:** A sorted list of the top 5 optimal banking partners, displaying the bank name, interest rate, and algorithmic match score.

---

## 📊 Data Sources
* **Model Training Data:** The Random Forest model was trained on a robust dataset of 500 historical loan records, featuring deep financial metrics like FICO scores, annual income, and past loan statuses.
* **Live Bank Data:** The matching engine utilizes a thoroughly cleaned, real-world dataset (`Banks-Interest-Rates_india.csv`) containing current interest rate slabs for major Indian banks across various loan tenures.

---

## 💻 Tech Stack
* **Backend:** Python, FastAPI, Uvicorn
* **Machine Learning:** Scikit-Learn (Random Forest), Pandas, NumPy
* **Frontend:** HTML5, CSS3, JavaScript, Bootstrap 5
* **Deployment:** Render (Cloud Hosting)

---

## 👥 The Team
This project was built collaboratively for HACKMAN held @DSCE by the following members:
* **[rithikh321](https://github.com/rithikh321)**
* **[Prakeerth1212](https://github.com/Prakeerth1212)** 
* **[DMANPRO](https://github.com/DMANPRO)**
