import pandas as pd                                  # Importing pandas for data manipulation and analysis
import numpy as np                                   # Importing numpy for numerical operations and handling blanks (NaN)

def engineer_application_features(df):
    """Engineer features from main application table"""
    
    df = df.copy()                                   # Creating an independent copy to protect the original data in memory

    # --- Fix DAYS_EMPLOYED anomaly ---
    df['DAYS_EMPLOYED_ANOMALY'] = (df['DAYS_EMPLOYED'] == 365243).astype(int)  # Flag '1' if applicant is a pensioner (365243 code)
    df["DAYS_EMPLOYED"] = df['DAYS_EMPLOYED'].replace(365243, np.nan)          # Replace 365243 with NaN to fix ML math distortion

    # --- Age and employment features ---
    df['AGE_YEARS'] = -df['DAYS_BIRTH'] / 365.25                               # Converting negative days to positive age in years
    df["EMPLOYED_YEARS"] = -df['DAYS_EMPLOYED'] / 365.25                       # Converting negative days to positive years worked
    df['PUBLISH_YEARS'] = -df['DAYS_ID_PUBLISH'] / 365.25                      # Converting negative days since ID publish to years
    df['LAST_PHONE_YEARS'] = -df['DAYS_LAST_PHONE_CHANGE'] / 365.25            # Converting negative days since phone change to years
    df['REGISTRATION_YEARS'] = -df['DAYS_REGISTRATION'] / 365.25               # Converting negative days since registration to years

    # --- Ratio features (important for credit risk) ---
    df['CREDIT_INCOME_RATIO'] = df['AMT_CREDIT'] / (df["AMT_INCOME_TOTAL"] + 1)   # Measuring loan size compared to income (+1 prevents dividing by zero)
    df["ANNUITY_INCOME_RATIO"] = df['AMT_ANNUITY'] / (df['AMT_INCOME_TOTAL'] + 1) # Measuring monthly payment burden relative to income
    df['CREDIT_GOODS_RATIO'] = df['AMT_CREDIT'] / (df['AMT_GOODS_PRICE'] + 1)     # Measuring loan amount against the price of the goods
    df['ANNUITY_CREDIT_RATIO'] = df['AMT_ANNUITY'] / (df['AMT_CREDIT'] + 1)       # Measuring monthly payment size relative to total loan
    df['EMPLOYED_TO_AGE_RATIO'] = df['EMPLOYED_YEARS'] / (df['AGE_YEARS'] + 0.001)# Measuring % of life spent working (+0.001 prevents dividing by zero)
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / (df['CNT_FAM_MEMBERS'] + 1)# Divide total income by family size for individual wealth
    df['CHILDREN_RATIO'] = df["CNT_CHILDREN"] / (df["CNT_FAM_MEMBERS"] + 1)       # Calculating ratio of children to total family members
    df["GOODS_CREDIT_DIFF"] = df['AMT_GOODS_PRICE'] - df['AMT_CREDIT']            # Calculating exact monetary difference between goods price and loan

    # --- External Credit Score Features ---
    ext = df[['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']]                   # Isolate the 3 external bureau scores into a mini-dataframe
    df['EXT_SOURCE_MEAN'] = ext.mean(axis=1)                                   # Calculate the average of the 3 scores horizontally
    df['EXT_SOURCE_SUM'] = ext.sum(axis=1)                                     # Add the 3 scores together
    df["EXT_SOURCE_PROD"] = ext.prod(axis=1)                                   # Multiply scores together (harshly penalizes any single low score)
    df['EXT_SOURCE_STD'] = ext.std(axis=1)                                     # Calculate standard deviation (measures score inconsistency)
    df['EXT_SOURCE_MIN'] = ext.min(axis=1)                                     # Find the single lowest (worst) score
    df['EXT_SOURCE_MAX'] = ext.max(axis=1)                                     # Find the single highest (best) score
    df['EXT_SOURCE_RANGE'] = df['EXT_SOURCE_MAX'] - df['EXT_SOURCE_MIN']       # Gap between best and worst score (large gaps indicate hidden risk)
    df['EXT_SOURCE_NANCOUNT'] = ext.isnull().sum(axis=1)                       # Count missing scores (identifies risky "thin-file" applicants)

    # --- Weighted combination ---
    # Creating custom score heavily favoring EXT_SOURCE_2 (50%), then 3 (30%), then 1 (20%). Fill blanks with 0. we can change according based on our preference but since EXT_SOURCE_3 has more -ve correlation with TARGET but still we chose EXT_SOURCE_2 becoz it has only 0.2% missing values but EXT_SOURCE_3 has 19%
    df['EXT_SOURCE_WEIGHTED'] = (df['EXT_SOURCE_1'].fillna(0) * 0.2 + 
                                 df['EXT_SOURCE_2'].fillna(0) * 0.5 + 
                                 df['EXT_SOURCE_3'].fillna(0) * 0.3)

    # --- Document submission count ---
    doc_cols = [c for c in df.columns if c.startswith("FLAG_DOCUMENT")]        # Finding all columns starting with "FLAG_DOCUMENT"
    df['TOTAL_DOCS_SUBMITTED'] = df[doc_cols].sum(axis=1)                      # Suming them horizontally (more documents = higher stability)

    # --- Contact flag count ---
    contact_cols = ['FLAG_MOBIL','FLAG_EMP_PHONE','FLAG_WORK_PHONE','FLAG_CONT_MOBILE','FLAG_PHONE','FLAG_EMAIL']
    df['TOTAL_CONTACTS'] = df[contact_cols].sum(axis=1)                        # Sum contact methods to create a "reachability" score

    # --- Social circle default rate features ---
    df["SOCIAL_CIRCLE_DEF_RATE"] = df['DEF_30_CNT_SOCIAL_CIRCLE'] / (df['OBS_30_CNT_SOCIAL_CIRCLE'] + 0.001) # % of social circle that defaulted

    # --- Income Outlier Cap ---
    income_99th = df['AMT_INCOME_TOTAL'].quantile(0.99)                        # Finding exact income representing the top 1% of earners
    df['AMT_INCOME_TOTAL'] = df['AMT_INCOME_TOTAL'].clip(upper=income_99th)    # Forcing extreme outliers down to this ceiling to fix math distortion

    # --- Hour and weekday of application ---
    df['IS_WEEKEND_APPLY'] = df['WEEKDAY_APPR_PROCESS_START'].isin(['SATURDAY','SUNDAY']).astype(int) # Flag '1' if applied on weekend
    df['IS_NIGHT_APPLY'] = df['HOUR_APPR_PROCESS_START'].between(22, 6).astype(int)                   # Flag '1' if applied late at night (10PM-6AM)

    # --- Apartment features median fill ---
    apt_cols = [c for c in df.columns if ('AVG' in c or 'MEDI' in c or "MODE" in c) and df[c].dtype != 'object'] # Finding numeric housing columns only using the attached word in the columns
    for col in apt_cols:                                                       # Looping through those housing columns
        df[col] = df[col].fillna(df[col].median())                             # Replacing blank rows with the median average of that column
        
    return df                                                                  # Sending the fully engineered dataset back to the main program


# --- Execution Block ---
if __name__ == "__main__":                                                     # Ensuring this only runs if executed directly (protects API imports)
    app_train = pd.read_csv('data/raw/application_train.csv')                  # Loading raw training data
    app_test = pd.read_csv('data/raw/application_test.csv')                    # Loading raw testing data
    
    app_train = engineer_application_features(app_train)                       # Passing training data through the engineering function
    app_test = engineer_application_features(app_test)                         # Passing testing data through the same function for consistency
    
    print(f"After application FE: {app_train.shape}")                          # Print final shape to confirm it worked on not








def aggregate_bureau(bureau_path, bureau_balance_path):
    """Aggregate bureau.csv and bureau_balance.csv to SK_ID_CURR level."""
    bureau = pd.read_csv(bureau_path)                                           #Defines the function and loads the two raw CSV files into memory.
    bureau_balance = pd.read_csv(bureau_balance_path)


    #STEP 1. Aggregate bureau_balance -->  bureau level
    bb_agg = bureau_balance.groupby("SK_ID_BUREAU").agg(BB_MONTHS_COUNT = ('MONTHS_BALANCE','count'),                                            #It groups all past loans in the same loan id becoz we cannot directly add all month data in table we need to summarize data
                                                        BB_STATUS_C_COUNT = ("STATUS", lambda x: (x=="C").sum()),                                #CLosed loan
                                                        BB_STATUS_C_COUNT = ("STATUS", lambda x: (x=="X").sum()),                                #Unknown loan
                                                        BB_DPD_COUNT = ('STATUS', lambda x: x.isin(["1",'2',"3",'4','5']).sum()),                #DPD means Days Past Due. here it counts how many total months the person was late
                                                        BB_DPD_MAX = ('STATUS', lambda x: max([int(s) for s in x if s.isdigit()], default =0)),  #calculate the absolute worst lateness category they ever hit (e.g., category '5' means 120+ days late or sold or written off)
    ).reset_index()

    bureau = bureau.merge(bb_agg, on ='SK_ID_BUREAU', how = 'left')                                                                              # Takes the 5 new columns I just created and glues them onto the right side of the main bureau table using the Loan ID as the anchor.



    #STEP 2. Seperate active vs closed credits
    active_bereau = bureau[bureau['CREDIT_ACTIVE'] == 'Active']
    closed_bereau = bureau[bureau['CREDIT_ACTIVE'] == 'Closed']

    #STEP 3. Full bureau aggregation
    bureau_agg = bureau.groupby('SK_ID_CURR').agg(
                                                #Count Features:
                                                B_LOAN_COUNT = ("SK_ID_BUREAU", 'count'),
                                                B_ACTIVE_COUNT = ('CREDIT_ACTIVE', lambda x : (x=='Active').sum()),
                                                B_CLOSED_COUNT = ('CREDIT_ACTIVE', lambda x : (x =='Closed').sum()),
                                                B_BAD_DEBT_COUNT = ('CREDIT_ACTIVE', lambda x : (x=='Bad debt').sum()),
                                                B_OVERDUE_COUNT = ('CREDIT_DAY_OVERDUE', lambda x : (x > 0).sum()),

                                                #Amount Features:
                                                B_AMT_CREDIT_SUM_MEAN = ('AMT_CREDIT_SUM', 'mean'),
                                                B_AMT_CREDIT_SUM_MAX = ('B_AMT_CREDIT_SUM', 'max'),
                                                B_AMT_CREDIT_SUM_TOTAL = ('B_AMT_CREDIT_SUM', 'sum'),
                                                B_AMT_DEBT_TOTAL = ('AMT_CREDIT_SUM_DEBT' , 'sum'),
                                                B_AMT_DEBT_MEAN = ('AMT_CREDIT_SUM_DEBT', 'mean'),
                                                B_AMT_OVERDUE_MAX = ('AMT_CREDIT_SUM_OVERDUE', 'max')    
                                                B_AMT_OVERDUE_MEAN = ('AMT_CREDIT_SUM_OVERDUE', 'mean'),   

                                                #Days OVerdue Features:
                                                B_DPD_MAX_MAX = ('CREDIT_DAY_OVERDUE','max'),
                                                B_DPD_MEAN = ('CREDIT_DAY_OVERDUE','mean'),

                                                #Credit Duration Features:
                                                B_FUTURE_COUNT =('DAYS_CREDIT_ENDDATE', lambda x:(x > 0).sum()),
                                                B_UPDATE_RECENCY_MEAN = ('DAYS_CREDIT_UPDATE','mean'),
                                                B_CREDIT_LENGTH_MEAN = ("DAYS_CREDIT"),'mean'),
    
                                                #Derived Ration Features:
                                                     

    























    )
