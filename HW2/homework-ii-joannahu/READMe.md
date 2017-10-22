HW2_Read me

Pingyuan Wang: pw2435
Qiong Hu: qh2174

This project is to use linear model to predict NYC house rental fee with data posted by the Census in 2014. The basic approach can be seen below. The R^2 is around 0.47 for the final selected model.

1. Choose observations with rent amount applicable 

2. Manually pick features with following principles:
	Select features related to house price, condition, location, etc
	Remove features related to householder, owners, etc

3. Remove biased features, use 0.9 as threshold. If a specific value in a feature is more than 90%, remove this feature. 

4. Data Processing
Unable to observe, not reported, not applicable all coded as missing value
	Fill categorical missing values with ¡®most frequent¡¯ strategy
	Fill numerical missing values with median value
	Remove feature if it has too many missing values (uf64, uf13, uf14)
	Remove observations that has more than 10 missing values
	Modify data as dummy variables

5. Model Building with 10-fold cross validation
	Linear regression and Ridge regression doesn¡¯t perform very well in this dataset, Ridge and linear regression give around 0.2 R^2 value.
	Our data is very high dimension and sparse, thus we decide to implement LASSO with grid search, found that optimal alpha is 2. With 10 CV, LASSO returns 0.47 R^2, which means the explanation power is 0.47. 
	
6. Model Evaluation
	Process test data set as we did for training set. Use the LASSO model to predict test data¡¯s response values. Then return predicted Y value. Make a plot to compare the predicted value and actual value. Except some extreme high values for actual y values, other values are very similar. The model is good!


