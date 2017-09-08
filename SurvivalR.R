#### Loading Libraries####
library(ggplot2) #loading visualizations
library(dplyr)
library(tidyr)
library(caret)
library(ranger) #trees and forests
library(ROCR)

###importing dataset
raw_data<-read.csv("//Vcn.ds.volvo.net/cli-hm/hm0226/a259842/My Documents/R/Projects/Survival Analysis On the Titanic DataSet/train.csv")
summary(raw_data)
str(raw_data)
#pclass Ticket class a proxy for socio-economic status (1 upper->3 lower class)

####setting a baseline of the raw performance####
baseline_Performance<-glm(Survived~Age+Sex+Pclass+Fare, family="binomial" , raw_data)
raw_data$pred<-0
raw_data$pred[!is.na(raw_data$Age)]<-predict(baseline_Performance, type="response")
#In sample misclassification
InSample_Result<-ifelse(raw_data$pred>0.5,1,0)
confusionMatrix<-table(InSample_Result,raw_data$Survived)

#Sensitivity = True positive/ (true positive +False Negative) 
confusionMatrix[1,1]/(confusionMatrix[1,1]+confusionMatrix[2,1])
0.87759599 #Sensitivity
#Accuracy= True positive + True Negative/ total observations
(confusionMatrix[1,1]+confusionMatrix[2,2])/891
0.7755331 #Accuracy
#Specificity= True negative/ (True negative+ False Positive)
confusionMatrix[2,2]/(confusionMatrix[2,2]+confusionMatrix[1,2])
0.6111111 #specificity
rm(confusionMatrix)
#Extra sample misclassification unable due to no labeled data -> cross validation or splitting of raw_data
test_data<-read.csv("//Vcn.ds.volvo.net/cli-hm/hm0226/a259842/My Documents/R/Projects/Survival Analysis On the Titanic DataSet/test.csv")
GainCurvePlot

####Early explanatory analysis ####

### Exploring visualy the dataset to check for multicollinearity, outliers and initial variance problems
### Although some issues are not necessarily for predicting outcomes it can provides us with additional insights

### missing values Analysis ###

missing_values_dataframe<-as.data.frame(lapply(raw_data, is.na))
missing_values_vector<-colSums(missing_values_dataframe)
#The variable age has 177 missing values. The question is are these missing at random (MAR) or non-random (MNAR) or missing completely at random MCAR
#We will use the package 'MICE: Multivariate Imputation by Chained Equations'
library(mice)
raw_data$missing_age<-ifelse(is.na(raw_data$Age), 0,1) #making a dummy variable and test whether there is a significant different between groups


### Check missing value pattern for survived
sum(raw_data$Survived[raw_data$missing_age==0])
#only 52  of 177 passenger with age missing survived #ratio 0.29
sum(raw_data$Survived[raw_data$missing_age==1])
#only 290  of 714 passenger with age missing survived #ratio 0.40

t.test(raw_data$Survived, raw_data$missing_age)
#check this again whether t test is the appropiate test for proportion between two proportion

### Check missing value pattern for fare
ggplot(raw_data, aes(log(raw_data$Fare)))+
  geom_histogram(aes(y=..density..))+
  geom_density()+
  facet_grid(~missing_age)+
  theme_classic()

ggplot(raw_data, aes((raw_data$Fare)))+
  geom_histogram(aes(y=..density..))+
  geom_density()+
  facet_grid(~missing_age)+
  theme_classic()

t.test(raw_data$Fare, raw_data$missing_age)

#use mice with random forest to impute values, we use random forest because it works a little better with categorical values and less normal, non-linear distributed values|source?
set.seed(1234) #to provide reproducible results
imputed.values<-mice.impute.rf(raw_data$Age, !is.na(raw_data$Age),raw_data[, c("Survived","Pclass", "Sex", "SibSp", "Parch", "Fare", "Embarked")])
mean(imputed.values) #30.21186
sd(imputed.values)   #12.64143
mean(raw_data$Age, na.rm=T) #29.69912
sd(raw_data$Age, na.rm=T)  #14.5265

#create new variable train_data with the substituted NA
train_data<-raw_data
impute<-function(x, y){ #x= missing values variable and y = the vector with the imputed values
k=0
 for (i in 1:length(x)){
   if (is.na(x[i])){
     k=k+1
     x[i]<-y[k]
     
   } 
 }
return(x)
}
train_data$Age<-impute(train_data$Age, imputed.values)
sum(is.na(train_data$Age)) #succesfull check
#for now we will drop the variables PassengerId, Name, Ticket , Cabin and missing_age
train_data<-train_data[,c("Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "pred")]

### Checking distribution of continiuous variables
histogram(train_data$Age) #more or less normal distributed
histogram(train_data$Fare) #very right skewed things with money often are
histogram(log(train_data$Fare)) #much better

#normal model
plot(train_data$Age, log(train_data$Fare))
abline(a=22.4120, b=0.327)
summary(lm(train_data$Fare ~ train_data$Age))
plot(lm(Fare~Age, train_data))

#log transformed model
plot(log(train_data$Fare), train_data$Age) #much better
abline(b=0.0007964, a=2.7149418)
train_data$logFare<-log(ifelse(train_data$Fare==0, 0.0001, train_data$Fare))
summary(lm(train_data$logFare ~ train_data$Age))
plot(lm(train_data$logFare ~ train_data$Age)) #very good we see little correlation. 
# we couldve also used partial regression by logistic regressing survived on logfare and then the residual on Age


###Exploring visualy the dataset to check for multicollinearity, outliers and initial variance problems
ggplot(train_data, aes(Survived, Age))+geom_jitter(aes(colour=Sex))
ggplot(train_data, aes(Survived, logFare))+geom_jitter(aes(colour=Sex))+ylim(0,10)
ggplot(train_data, aes(Survived,logFare))+geom_boxplot(aes(group=Survived))
#### Conclusions of the EDA ####

#### Machine learning algorithmes ####

###logistic regression with K-fold and leave-one-out cross validation


survival.glm<-glm(Survived~Pclass+Sex+Age+Fare+Embarked, data=train_data, family = binomial)
cv.classification<-cv.glm(train_data,survival.glm)
1-cv.classification$delta[1]
survival.glm
survival.glm.pred<-predict(survival.glm, type="response" )
survival.glm.pred<-ifelse(survival.glm.pred<0.5,0,1)
confusionMatrix<-table(survival.glm.pred, train_data$Survived)
confusionMatrix[1,1]/(confusionMatrix[1,1]+confusionMatrix[2,1])

### naive bayesian regression

### random forest
### gradient boosting of decision trees
### support vector machine learning





