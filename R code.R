setwd('C:/Users/user/Desktop/kakku')

data = read.csv('train.csv')
target = read.csv('test.csv')

summary(data$ID_code)
str(data$ID_code)
colnames(data)

data$target = as.factor(data$target)
data = subset(data, select = -c(ID_code))

#Missing Value Analysis
sum(is.na(data))


#Feature Selection
library(caret)
set.seed(100)
options(warn=-1)

ctrl <- rfeControl(functions = lmFuncs,
                   method = "repeatedcv",
                   repeats = 5,
                   verbose = FALSE)

lmProfile <- rfe(x=data[,-1], y=data$target, rfeControl = ctrl)

lmProfile

df = data[,c("target","var_68", "var_12", "var_148", "var_108", "var_91")]

#Outlier Analysis
for(i in colnames(df[,-1])){
  print(i)
  val = df[,i][df[,i]%in%boxplot.stats(df[,i])$out]
  df = df[which(!df[,i]%in%val),]
}


#Splitting into test and train
library('caret')
set.seed(1234)
train.index = createDataPartition(df$target,p=0.80,list=FALSE)
train = df[train.index,]
test = df[-train.index,]



#Decision Tree
library(rpart)
library('e1071')
dt_model = rpart(target~., data = train, method = 'class')

dt_predict = predict(dt_model, test[,-1], type = 'class')

confusionMatrix(dt_predict, test[,1])
#Accuracy = 0.8995
#precision = 0.899
#Recall = 1.00


#Random Forest
library(randomForest)
rf_model = randomForest(target~., train, importance = TRUE, ntree = 100)

rf_predict = predict(rf_model, test[,-1])

confusionMatrix(rf_predict, test[,1])
#Accuracy = 0.8989
#precision = 0.899
#Recall = 0.998



#Knn Model
library(class)
knn_model = knn(train[,-1], test[,-1], cl= train$target, k=7)

confusionMatrix(knn_model, test[,1])
#Accuracy = 0.895
#precision = 0.900
#Recall = 0.994



#Logistic Regression
lr_model = glm(target~., data = train, family = 'binomial')

lr_predict = predict(lr_model, test[,-1])

lr_predict = ifelse(lr_predict>0.5,1,0)
lr_predict = as.factor(lr_predict)

confusionMatrix(lr_predict, test[,1])
#Accuracy = 0.8995
#precision = 0.899
#Recall = 1


target = target[,c("var_68", "var_12", "var_148", "var_108", "var_91")]
target$target = predict(rf_model,target)

write.csv(target, 'target.csv')