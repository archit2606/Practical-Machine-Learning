---
output:
        html_document:
                keep_md: true
---
Practical Machine Learning Course Project
============================================

## Overview

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it.

In this project, we will use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways.

The data consists of a Training data and a Test data (to be used to validate the selected model).

The goal of your project is to predict the manner in which they did the exercise. This is the “classe” variable in the training set. You may use any of the other variables to predict with.

## Loading data

The following libraries will be used in the following project.


```r
library(caret)
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```r
library(randomForest)
```

```
## randomForest 4.6-14
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

```r
library(ggplot2)
library(rpart)
library(rattle)
```

```
## Rattle: A free graphical interface for data science with R.
## Version 5.3.0 Copyright (c) 2006-2018 Togaware Pty Ltd.
## Type 'rattle()' to shake, rattle, and roll your data.
```

```
## 
## Attaching package: 'rattle'
```

```
## The following object is masked from 'package:randomForest':
## 
##     importance
```

```r
library(gbm)
```

```
## Loaded gbm 2.1.5
```

```r
library(rpart.plot)
library(corrplot)
```

```
## corrplot 0.84 loaded
```

Loading data into traing and testing dataset.


```r
training <- read.csv("pml-training.csv")
testing <- read.csv("pml-testing.csv")
dim(training)
```

```
## [1] 19622   160
```

```r
dim(testing)
```

```
## [1]  20 160
```

## Cleaning datasets

Observations with NA values will be removed in this step.


```r
sum(complete.cases(training))
```

```
## [1] 406
```

Assigning values 0 to NA values.


```r
training <- training[, colSums(is.na(training)) == 0] 
testing <- testing[, colSums(is.na(testing)) == 0] 
```

Get rid of columns that don't contribute too much in the measurements.


```r
classe <- training$classe
trainRemove <- grepl("^X|timestamp|window", names(training))
training <- training[, !trainRemove]
trainCleaned <- training[, sapply(training, is.numeric)]
trainCleaned$classe <- classe
testRemove <- grepl("^X|timestamp|window", names(testing))
testing <- testing[, !testRemove]
testCleaned <- testing[, sapply(testing, is.numeric)]
```

The data is cleaned now and it does not contain any  NA values.

Splitting the cleaned data into training and testing data sets.

## Diving the data into training and testing set


```r
traina <- createDataPartition(trainCleaned$classe, p=0.70, list=F)
trainData <- trainCleaned[traina, ]
testData <- trainCleaned[-traina, ]
```

## Training data

First fit the data on the model using random forest which will identify the important variables. Also, we will use the 5 fold cross validation in training the dataset.


```r
controlRf <- trainControl(method="cv", 5)
fit <- train(classe ~ ., data=trainData, method="rf", trControl=controlRf, ntree=250)
fit
```

```
## Random Forest 
## 
## 13737 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 10988, 10991, 10989, 10990, 10990 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##    2    0.9910464  0.9886728
##   27    0.9905367  0.9880292
##   52    0.9863876  0.9827798
## 
## Accuracy was used to select the optimal model using the largest value.
## The final value used for the model was mtry = 2.
```


```r
predicta <- predict(fit, testData)
confusionMatrix(testData$classe, predicta)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1672    1    0    0    1
##          B   14 1122    3    0    0
##          C    0    5 1020    1    0
##          D    0    0   14  947    3
##          E    0    0    0    4 1078
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9922          
##                  95% CI : (0.9896, 0.9943)
##     No Information Rate : 0.2865          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9901          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9917   0.9947   0.9836   0.9947   0.9963
## Specificity            0.9995   0.9964   0.9988   0.9966   0.9992
## Pos Pred Value         0.9988   0.9851   0.9942   0.9824   0.9963
## Neg Pred Value         0.9967   0.9987   0.9965   0.9990   0.9992
## Prevalence             0.2865   0.1917   0.1762   0.1618   0.1839
## Detection Rate         0.2841   0.1907   0.1733   0.1609   0.1832
## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
## Balanced Accuracy      0.9956   0.9956   0.9912   0.9957   0.9977
```

## Checking the performance of the model

From the selected model let's check the performance.


```r
accuracy <- postResample(predicta, testData$classe)
accuracy
```

```
##  Accuracy     Kappa 
## 0.9921835 0.9901106
```


```r
outofsamplerror <- 1 - as.numeric(confusionMatrix(testData$classe, predicta)$overall[1])
outofsamplerror
```

```
## [1] 0.007816483
```

From the above final model the selected model was mtry = 2, with accuracy 99% and the out of sample error is less than 1%. so, we can select this model and predict the values for the rest of the data.

## Prediction for Testing dataset

In this step we are applyig the final model to the original dataset to get the result for the 20 cases.


```r
result <- predict(fit, testCleaned[, -length(names(testCleaned))])
result
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

## Appendix : Figures

1) Correlation Matrix Validation


```r
corrPlot <- cor(trainData[, -length(names(trainData))])
corrplot(corrPlot, method="color")
```

![](https://github.com/archit2606/Practical-Machine-Learning/blob/master/correlation%20matrix.png)<!-- -->

2) Decision tree 


```r
predtree <- rpart(classe ~ ., data = trainData, method = "class")
prp(predtree)
```

![](https://github.com/archit2606/Practical-Machine-Learning/blob/master/prediction%20tree.png)<!-- -->
