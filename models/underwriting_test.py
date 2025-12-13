# =========================================================
# FILE 1: loan_approval_predictor.py
# =========================================================
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class LoanApprovalPredictor:
    def __init__(self, model_path='underwriting_model_regularized.pkl'):
        """Initialize the loan approval predictor"""
        self.model = joblib.load(model_path)
        with open('model_info_regularized.json', 'r') as f:
            self.model_info = json.load(f)
        
        # Feature lists from training
        self.numerical_features = [
            'age', 'years_employed', 'annual_income', 'existing_loan_balance',
            'existing_emi_monthly', 'credit_score', 'payment_history_default',
            'credit_inquiry_last_6m', 'num_open_accounts', 'property_value',
            'requested_loan_amount', 'requested_loan_tenure', 'debt_to_income_ratio',
            'loan_to_income_ratio', 'credit_age_months', 'income_to_existing_debt',
            'credit_utilization', 'employment_stability'
        ]
        
        self.categorical_features = [
            'gender', 'city', 'employment_type', 'education_level',
            'marital_status', 'home_ownership', 'property_type'
        ]
        
        print("="*60)
        print("LOAN APPROVAL PREDICTOR - INITIALIZED")
        print(f"Model: {self.model_info['model_name']}")
        print(f"Test F1 Score: {self.model_info['test_f1']:.4f}")
        print("="*60)
    
    def create_features(self, df):
        """Create engineered features for prediction"""
        df = df.copy()
        
        # Create the same features used during training
        df['income_to_existing_debt'] = df['annual_income'] / (df['existing_loan_balance'] + 1)
        df['credit_utilization'] = df['existing_loan_balance'] / (df['annual_income'] + 1)
        df['employment_stability'] = df['years_employed'] / (df['age'] - 18 + 1)
        
        # Calculate monthly income if not provided
        if 'monthly_income' not in df.columns and 'annual_income' in df.columns:
            df['monthly_income'] = df['annual_income'] / 12
        
        # Calculate EMI if not provided (using simplified formula)
        if 'estimated_monthly_emi' not in df.columns:
            # Simple EMI calculation: P*r*(1+r)^n / ((1+r)^n - 1)
            # Assuming 10% interest rate for estimation
            interest_rate = 0.10 / 12  # Monthly interest rate
            principal = df['requested_loan_amount']
            tenure_months = df['requested_loan_tenure']
            
            df['estimated_monthly_emi'] = principal * interest_rate * (1 + interest_rate)**tenure_months / \
                                         ((1 + interest_rate)**tenure_months - 1)
        
        return df
    
    def predict(self, customer_data):
        """Make prediction for customer data"""
        # Convert to DataFrame if not already
        if isinstance(customer_data, dict):
            customer_data = pd.DataFrame([customer_data])
        
        # Create features
        customer_data = self.create_features(customer_data)
        
        # Prepare features for model
        X_new = customer_data[self.numerical_features + self.categorical_features]
        
        # Get prediction and probabilities
        probabilities = self.model.predict_proba(X_new)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        
        # Generate detailed results
        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            customer = customer_data.iloc[i]
            result = self._generate_detailed_result(customer, pred, prob)
            results.append(result)
        
        return results[0] if len(results) == 1 else results
    
    def _generate_detailed_result(self, customer, prediction, probability):
        """Generate detailed prediction result with explanations"""
        
        result = {
            'customer_id': customer.get('customer_id', 'USER001'),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'approval_status': 'Approved' if prediction == 1 else 'Rejected',
            'approval_probability': float(probability),
            'confidence_level': self._get_confidence_level(probability),
            'decision': 'APPROVE' if prediction == 1 else 'REJECT',
            'key_factors': [],
            'recommendations': [],
            'risk_indicators': [],
            'strength_indicators': []
        }
        
        # Analyze key factors
        self._analyze_factors(customer, result)
        
        # Generate recommendations
        self._generate_recommendations(customer, result)
        
        # Calculate risk score
        result['risk_score'] = self._calculate_risk_score(customer, result)
        
        return result
    
    def _get_confidence_level(self, probability):
        """Determine confidence level based on probability"""
        if probability > 0.8 or probability < 0.2:
            return "High"
        elif probability > 0.7 or probability < 0.3:
            return "Medium"
        else:
            return "Low"
    
    def _analyze_factors(self, customer, result):
        """Analyze key factors affecting the decision"""
        
        # Positive factors
        if customer['credit_score'] >= 750:
            result['strength_indicators'].append(f"Excellent credit score: {customer['credit_score']}")
        
        if customer['years_employed'] >= 5:
            result['strength_indicators'].append(f"Stable employment: {customer['years_employed']} years")
        
        if customer['annual_income'] / customer['requested_loan_amount'] >= 2:
            result['strength_indicators'].append("Healthy income-to-loan ratio")
        
        # Risk factors
        if customer['payment_history_default'] > 2:
            result['risk_indicators'].append(f"Payment defaults: {customer['payment_history_default']}")
        
        if customer['debt_to_income_ratio'] > 40:
            result['risk_indicators'].append(f"High debt-to-income ratio: {customer['debt_to_income_ratio']:.1f}%")
        
        if customer['credit_inquiry_last_6m'] > 4:
            result['risk_indicators'].append(f"Multiple recent credit inquiries: {customer['credit_inquiry_last_6m']}")
        
        # Key decision factors
        if customer['credit_score'] < 650:
            result['key_factors'].append("Low credit score")
        elif customer['credit_score'] > 750:
            result['key_factors'].append("High credit score")
        
        if customer['employment_stability'] > 0.3:
            result['key_factors'].append("Good employment stability")
    
    def _generate_recommendations(self, customer, result):
        """Generate personalized recommendations"""
        
        if result['decision'] == 'REJECT':
            if customer['credit_score'] < 650:
                result['recommendations'].append("Improve credit score to at least 650")
            
            if customer['debt_to_income_ratio'] > 40:
                result['recommendations'].append("Reduce existing debt to lower DTI ratio")
            
            if customer['payment_history_default'] > 0:
                result['recommendations'].append("Make timely payments for next 6 months")
            
            result['recommendations'].append("Consider a smaller loan amount")
            result['recommendations'].append("Reapply after 3-6 months with improved profile")
        
        else:  # APPROVED
            if customer['credit_score'] < 700:
                result['recommendations'].append("Maintain or improve credit score for better rates")
            
            result['recommendations'].append("Ensure timely EMI payments")
            result['recommendations'].append("Maintain stable employment")
            result['recommendations'].append("Avoid new credit applications for 6 months")
    
    def _calculate_risk_score(self, customer, result):
        """Calculate a comprehensive risk score (0-100, lower is better)"""
        risk_score = 50  # Base score
        
        # Adjust based on factors
        if customer['credit_score'] < 600:
            risk_score += 30
        elif customer['credit_score'] < 650:
            risk_score += 20
        elif customer['credit_score'] < 700:
            risk_score += 10
        elif customer['credit_score'] >= 750:
            risk_score -= 15
        
        if customer['payment_history_default'] > 0:
            risk_score += customer['payment_history_default'] * 5
        
        if customer['debt_to_income_ratio'] > 40:
            risk_score += (customer['debt_to_income_ratio'] - 40) * 0.5
        
        if customer['years_employed'] < 2:
            risk_score += 10
        
        # Normalize to 0-100
        risk_score = max(0, min(100, risk_score))
        
        return int(risk_score)
    
    def batch_predict(self, customers_data):
        """Make predictions for multiple customers"""
        if isinstance(customers_data, dict):
            customers_data = pd.DataFrame([customers_data])
        
        results = []
        for _, customer in customers_data.iterrows():
            result = self.predict(customer.to_dict())
            results.append(result)
        
        return results

# =========================================================
# FILE 2: loan_test_interface.py (User Interface)
# =========================================================
class LoanTestInterface:
    def __init__(self):
        """Initialize the testing interface"""
        self.predictor = LoanApprovalPredictor()
        self.demo_profiles = self._create_demo_profiles()
    
    def _create_demo_profiles(self):
        """Create sample customer profiles for demonstration"""
        return {
            '1': {
                'name': 'Ideal Borrower',
                'description': 'High income, excellent credit, stable job',
                'data': {
                    'age': 35,
                    'gender': 'M',
                    'city': 'Bangalore',
                    'employment_type': 'Salaried',
                    'years_employed': 8,
                    'annual_income': 1500000,
                    'existing_loan_balance': 200000,
                    'existing_emi_monthly': 5000,
                    'credit_score': 810,
                    'payment_history_default': 0,
                    'credit_inquiry_last_6m': 1,
                    'num_open_accounts': 3,
                    'num_delinquent_accounts': 0,
                    'education_level': 'Post-Graduate',
                    'marital_status': 'Married',
                    'home_ownership': 'Owned',
                    'property_type': 'Apartment',
                    'property_value': 8000000,
                    'requested_loan_amount': 3000000,
                    'requested_loan_tenure': 36,
                    'debt_to_income_ratio': 15.0,
                    'loan_to_income_ratio': 200.0,
                    'credit_age_months': 120
                }
            },
            '2': {
                'name': 'Risky Borrower',
                'description': 'Low credit score, high debt, unstable employment',
                'data': {
                    'age': 25,
                    'gender': 'F',
                    'city': 'Delhi',
                    'employment_type': 'Self-Employed',
                    'years_employed': 1,
                    'annual_income': 600000,
                    'existing_loan_balance': 500000,
                    'existing_emi_monthly': 15000,
                    'credit_score': 580,
                    'payment_history_default': 3,
                    'credit_inquiry_last_6m': 6,
                    'num_open_accounts': 7,
                    'num_delinquent_accounts': 1,
                    'education_level': 'Graduate',
                    'marital_status': 'Single',
                    'home_ownership': 'Rented',
                    'property_type': 'Apartment',
                    'property_value': 0,
                    'requested_loan_amount': 1000000,
                    'requested_loan_tenure': 24,
                    'debt_to_income_ratio': 65.0,
                    'loan_to_income_ratio': 166.7,
                    'credit_age_months': 24
                }
            },
            '3': {
                'name': 'Average Borrower',
                'description': 'Moderate profile with some strengths and weaknesses',
                'data': {
                    'age': 42,
                    'gender': 'M',
                    'city': 'Mumbai',
                    'employment_type': 'Business Owner',
                    'years_employed': 5,
                    'annual_income': 1200000,
                    'existing_loan_balance': 800000,
                    'existing_emi_monthly': 20000,
                    'credit_score': 720,
                    'payment_history_default': 1,
                    'credit_inquiry_last_6m': 3,
                    'num_open_accounts': 5,
                    'num_delinquent_accounts': 0,
                    'education_level': 'Graduate',
                    'marital_status': 'Married',
                    'home_ownership': 'Mortgaged',
                    'property_type': 'House',
                    'property_value': 5000000,
                    'requested_loan_amount': 2000000,
                    'requested_loan_tenure': 48,
                    'debt_to_income_ratio': 25.0,
                    'loan_to_income_ratio': 166.7,
                    'credit_age_months': 96
                }
            }
        }
    
    def display_menu(self):
        """Display main menu"""
        print("\n" + "="*60)
        print("PERSONAL LOAN APPROVAL PREDICTOR")
        print("="*60)
        print("\nMAIN MENU:")
        print("1. Test with your own details")
        print("2. View demo profiles")
        print("3. Test a demo profile")
        print("4. Batch test multiple scenarios")
        print("5. View model information")
        print("6. Exit")
    
    def get_user_input(self):
        """Get loan application details from user"""
        print("\n" + "="*60)
        print("ENTER YOUR LOAN APPLICATION DETAILS")
        print("="*60)
        
        customer_data = {}
        
        # Personal Information
        print("\n--- PERSONAL INFORMATION ---")
        customer_data['age'] = int(input("Age: "))
        customer_data['gender'] = input("Gender (M/F): ").upper()
        customer_data['city'] = input("City: ")
        customer_data['marital_status'] = input("Marital Status (Single/Married/Divorced/Widowed): ")
        customer_data['education_level'] = input("Education Level (12th/Diploma/Graduate/Post-Graduate): ")
        
        # Employment Information
        print("\n--- EMPLOYMENT INFORMATION ---")
        customer_data['employment_type'] = input("Employment Type (Salaried/Self-Employed/Business Owner): ")
        customer_data['years_employed'] = float(input("Years at current employment: "))
        customer_data['annual_income'] = float(input("Annual Income (in INR): "))
        
        # Financial Information
        print("\n--- FINANCIAL INFORMATION ---")
        customer_data['existing_loan_balance'] = float(input("Existing Loan Balance (in INR): "))
        customer_data['existing_emi_monthly'] = float(input("Existing Monthly EMI (in INR): "))
        customer_data['property_value'] = float(input("Property Value (if any, in INR): "))
        customer_data['home_ownership'] = input("Home Ownership (Owned/Rented/Mortgaged): ")
        customer_data['property_type'] = input("Property Type (House/Apartment/Villa/Studio): ")
        
        # Credit Information
        print("\n--- CREDIT INFORMATION ---")
        customer_data['credit_score'] = int(input("Credit Score (300-900): "))
        customer_data['cibil_score'] = int(input("CIBIL Score (300-900, or same as credit): ") or customer_data['credit_score'])
        customer_data['payment_history_default'] = int(input("Number of payment defaults (0-10): "))
        customer_data['credit_inquiry_last_6m'] = int(input("Credit inquiries in last 6 months: "))
        customer_data['num_open_accounts'] = int(input("Number of open credit accounts: "))
        customer_data['num_delinquent_accounts'] = int(input("Number of delinquent accounts: "))
        customer_data['credit_age_months'] = int(input("Credit history age in months: "))
        
        # Loan Request
        print("\n--- LOAN REQUEST ---")
        customer_data['requested_loan_amount'] = float(input("Requested Loan Amount (in INR): "))
        customer_data['requested_loan_tenure'] = int(input("Loan Tenure in months (12/24/36/48/60): "))
        customer_data['debt_to_income_ratio'] = float(input("Debt-to-Income Ratio (%): "))
        customer_data['loan_to_income_ratio'] = float(input("Loan-to-Income Ratio (%): "))
        
        # Add a customer ID
        customer_data['customer_id'] = f"USER_{datetime.now().strftime('%H%M%S')}"
        
        return customer_data
    
    def display_demo_profiles(self):
        """Display available demo profiles"""
        print("\n" + "="*60)
        print("DEMO PROFILES")
        print("="*60)
        
        for key, profile in self.demo_profiles.items():
            print(f"\n[{key}] {profile['name']}")
            print(f"   {profile['description']}")
            print(f"   Key Stats:")
            print(f"   - Age: {profile['data']['age']}")
            print(f"   - Annual Income: ₹{profile['data']['annual_income']:,}")
            print(f"   - Credit Score: {profile['data']['credit_score']}")
            print(f"   - Requested Loan: ₹{profile['data']['requested_loan_amount']:,}")
    
    def test_demo_profile(self, profile_key):
        """Test a demo profile"""
        if profile_key in self.demo_profiles:
            profile = self.demo_profiles[profile_key]
            print(f"\nTesting: {profile['name']}")
            print(f"Description: {profile['description']}")
            
            result = self.predictor.predict(profile['data'])
            self.display_result(result)
        else:
            print("Invalid profile selection!")
    
    def display_result(self, result):
        """Display prediction result in a user-friendly format"""
        print("\n" + "="*60)
        print("LOAN APPROVAL DECISION")
        print("="*60)
        
        # Decision Banner
        if result['decision'] == 'APPROVE':
            print("\n✅ " + "="*30 + " ✅")
            print("✅       LOAN APPROVED!       ✅")
            print("✅ " + "="*30 + " ✅")
        else:
            print("\n❌ " + "="*30 + " ❌")
            print("❌       LOAN REJECTED       ❌")
            print("❌ " + "="*30 + " ❌")
        
        # Basic Information
        print(f"\nCustomer ID: {result['customer_id']}")
        print(f"Timestamp: {result['timestamp']}")
        print(f"Approval Probability: {result['approval_probability']:.1%}")
        print(f"Confidence Level: {result['confidence_level']}")
        print(f"Risk Score: {result['risk_score']}/100")
        
        # Key Factors
        if result['key_factors']:
            print(f"\n📊 KEY DECISION FACTORS:")
            for factor in result['key_factors']:
                print(f"   • {factor}")
        
        # Strength Indicators
        if result['strength_indicators']:
            print(f"\n✅ STRENGTHS:")
            for strength in result['strength_indicators']:
                print(f"   ✓ {strength}")
        
        # Risk Indicators
        if result['risk_indicators']:
            print(f"\n⚠️  RISK INDICATORS:")
            for risk in result['risk_indicators']:
                print(f"   • {risk}")
        
        # Recommendations
        if result['recommendations']:
            print(f"\n💡 RECOMMENDATIONS:")
            for i, rec in enumerate(result['recommendations'], 1):
                print(f"   {i}. {rec}")
        
        print("\n" + "="*60)
        
        # Save to file
        self.save_result(result)
    
    def save_result(self, result):
        """Save result to a text file"""
        filename = f"loan_decision_{result['customer_id']}_{datetime.now().strftime('%Y%m%d')}.txt"
        
        with open(filename, 'w') as f:
            f.write("="*60 + "\n")
            f.write("LOAN APPROVAL DECISION REPORT\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Customer ID: {result['customer_id']}\n")
            f.write(f"Timestamp: {result['timestamp']}\n")
            f.write(f"Decision: {result['decision']}\n")
            f.write(f"Approval Probability: {result['approval_probability']:.1%}\n")
            f.write(f"Confidence Level: {result['confidence_level']}\n")
            f.write(f"Risk Score: {result['risk_score']}/100\n\n")
            
            if result['key_factors']:
                f.write("Key Decision Factors:\n")
                for factor in result['key_factors']:
                    f.write(f"- {factor}\n")
                f.write("\n")
            
            if result['strength_indicators']:
                f.write("Strengths:\n")
                for strength in result['strength_indicators']:
                    f.write(f"+ {strength}\n")
                f.write("\n")
            
            if result['risk_indicators']:
                f.write("Risk Indicators:\n")
                for risk in result['risk_indicators']:
                    f.write(f"- {risk}\n")
                f.write("\n")
            
            if result['recommendations']:
                f.write("Recommendations:\n")
                for i, rec in enumerate(result['recommendations'], 1):
                    f.write(f"{i}. {rec}\n")
        
        print(f"Result saved to: {filename}")
    
    def batch_test(self):
        """Test multiple scenarios at once"""
        print("\n" + "="*60)
        print("BATCH TESTING")
        print("="*60)
        
        # Test all demo profiles
        results = []
        for key, profile in self.demo_profiles.items():
            result = self.predictor.predict(profile['data'])
            results.append({
                'profile': profile['name'],
                'decision': result['decision'],
                'probability': result['approval_probability'],
                'risk_score': result['risk_score']
            })
        
        # Display batch results
        print("\nBATCH TEST RESULTS:")
        print("-"*60)
        print(f"{'Profile':<20} {'Decision':<10} {'Probability':<15} {'Risk Score':<10}")
        print("-"*60)
        
        for res in results:
            decision_symbol = "✅" if res['decision'] == 'APPROVE' else "❌"
            print(f"{res['profile']:<20} {decision_symbol} {res['decision']:<8} {res['probability']:<14.1%} {res['risk_score']:<10}")
    
    def display_model_info(self):
        """Display model information"""
        print("\n" + "="*60)
        print("MODEL INFORMATION")
        print("="*60)
        
        info = self.predictor.model_info
        print(f"Model Name: {info['model_name']}")
        print(f"Training F1 Score: {info['train_f1']:.4f}")
        print(f"Testing F1 Score: {info['test_f1']:.4f}")
        print(f"Overfitting Gap: {info['overfitting_gap']:.4f}")
        print(f"CV Mean F1: {info['cv_mean_f1']:.4f}")
        print(f"CV Std F1: {info['cv_std_f1']:.4f}")
        print(f"Regularization Strength: C={info['regularization_strength']}")
        
        print(f"\nTop 10 Features Used:")
        for i, feature in enumerate(info['selected_features'][:10], 1):
            print(f"{i:2}. {feature}")
    
    def run(self):
        """Main execution loop"""
        while True:
            self.display_menu()
            choice = input("\nEnter your choice (1-6): ")
            
            if choice == '1':
                # Test with user input
                try:
                    customer_data = self.get_user_input()
                    result = self.predictor.predict(customer_data)
                    self.display_result(result)
                except Exception as e:
                    print(f"Error: {e}. Please check your inputs.")
            
            elif choice == '2':
                # View demo profiles
                self.display_demo_profiles()
            
            elif choice == '3':
                # Test a demo profile
                self.display_demo_profiles()
                profile_key = input("\nEnter profile number to test (1-3): ")
                self.test_demo_profile(profile_key)
            
            elif choice == '4':
                # Batch test
                self.batch_test()
            
            elif choice == '5':
                # Model information
                self.display_model_info()
            
            elif choice == '6':
                print("\nThank you for using Loan Approval Predictor!")
                break
            
            else:
                print("Invalid choice! Please try again.")
            
            input("\nPress Enter to continue...")

# =========================================================
# FILE 3: quick_test.py (For quick testing)
# =========================================================
def quick_test():
    """Quick test function for direct use"""
    
    # Your personal details - EDIT THESE VALUES
    my_details = {
        'age': 30,  # Your age
        'gender': 'M',  # M or F
        'city': 'Bangalore',  # Your city
        'employment_type': 'Salaried',  # Salaried/Self-Employed/Business Owner
        'years_employed': 5,  # Years in current job
        'annual_income': 1200000,  # Your annual income in INR
        'existing_loan_balance': 300000,  # Any existing loans
        'existing_emi_monthly': 10000,  # Monthly EMI payments
        'credit_score': 750,  # Your credit score (300-900)
        'cibil_score': 750,  # Your CIBIL score
        'payment_history_default': 0,  # Number of payment defaults
        'credit_inquiry_last_6m': 2,  # Credit checks in last 6 months
        'num_open_accounts': 3,  # Open credit accounts
        'num_delinquent_accounts': 0,  # Delinquent accounts
        'education_level': 'Graduate',  # 12th/Diploma/Graduate/Post-Graduate
        'marital_status': 'Single',  # Single/Married/Divorced/Widowed
        'home_ownership': 'Rented',  # Owned/Rented/Mortgaged
        'property_type': 'Apartment',  # House/Apartment/Villa/Studio
        'property_value': 0,  # Value of property if any
        'requested_loan_amount': 500000,  # Loan amount you want
        'requested_loan_tenure': 24,  # Loan period in months
        'debt_to_income_ratio': 15.0,  # Your DTI ratio in %
        'loan_to_income_ratio': 41.7,  # (Loan/Income)*100
        'credit_age_months': 60  # Credit history in months
    }
    
    # Initialize predictor
    predictor = LoanApprovalPredictor()
    
    # Make prediction
    print("\n" + "="*60)
    print("QUICK LOAN APPROVAL TEST")
    print("="*60)
    
    print("\nYour Details:")
    print(f"- Age: {my_details['age']}")
    print(f"- Annual Income: ₹{my_details['annual_income']:,}")
    print(f"- Credit Score: {my_details['credit_score']}")
    print(f"- Requested Loan: ₹{my_details['requested_loan_amount']:,}")
    print(f"- Loan Tenure: {my_details['requested_loan_tenure']} months")
    
    print("\n" + "-"*60)
    print("Processing your application...")
    
    result = predictor.predict(my_details)
    
    # Simple display
    print("\n" + "="*60)
    if result['decision'] == 'APPROVE':
        print("✅ LOAN APPROVED! ✅")
    else:
        print("❌ LOAN REJECTED ❌")
    
    print(f"\nApproval Probability: {result['approval_probability']:.1%}")
    print(f"Confidence: {result['confidence_level']}")
    print(f"Risk Score: {result['risk_score']}/100")
    
    if result['key_factors']:
        print(f"\nKey Factors:")
        for factor in result['key_factors']:
            print(f"• {factor}")
    
    if result['recommendations']:
        print(f"\nRecommendations:")
        for rec in result['recommendations']:
            print(f"• {rec}")
    
    print("\n" + "="*60)

# =========================================================
# MAIN EXECUTION
# =========================================================
if __name__ == "__main__":
    print("Loan Approval Testing System")
    print("="*60)
    
    # Choose mode
    print("\nSelect Mode:")
    print("1. Full Interactive Interface")
    print("2. Quick Test (Edit code with your details)")
    print("3. Test with Sample Data")
    
    mode = input("\nEnter mode (1-3): ")
    
    if mode == '1':
        # Run full interface
        interface = LoanTestInterface()
        interface.run()
    
    elif mode == '2':
        # Run quick test - EDIT the quick_test() function above with your details
        quick_test()
    
    elif mode == '3':
        # Test with sample data
        predictor = LoanApprovalPredictor()
        
        # Sample data
        sample_data = {
            'age': 35,
            'gender': 'M',
            'city': 'Mumbai',
            'employment_type': 'Salaried',
            'years_employed': 7,
            'annual_income': 1800000,
            'existing_loan_balance': 500000,
            'existing_emi_monthly': 15000,
            'credit_score': 780,
            'cibil_score': 780,
            'payment_history_default': 0,
            'credit_inquiry_last_6m': 1,
            'num_open_accounts': 2,
            'num_delinquent_accounts': 0,
            'education_level': 'Post-Graduate',
            'marital_status': 'Married',
            'home_ownership': 'Owned',
            'property_type': 'Apartment',
            'property_value': 10000000,
            'requested_loan_amount': 3000000,
            'requested_loan_tenure': 36,
            'debt_to_income_ratio': 12.5,
            'loan_to_income_ratio': 166.7,
            'credit_age_months': 84
        }
        
        result = predictor.predict(sample_data)
        
        print("\nSample Test Result:")
        print(f"Decision: {result['decision']}")
        print(f"Probability: {result['approval_probability']:.1%}")
        print(f"Risk Score: {result['risk_score']}/100")
    
    else:
        print("Invalid mode selected!")
