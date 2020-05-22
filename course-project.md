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
## Summary of sample sizes: 10988, 10990, 10990, 10990, 10990 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##    2    0.9901724  0.9875677
##   27    0.9898082  0.9871079
##   52    0.9800534  0.9747665
## 
## Accuracy was used to select the optimal model using the largest value.
## The final value used for the model was mtry = 2.
```

From the above final model the selected model was mtry = 2, with accuracy 99.1% and kappa 98.9%. so, we can select this model and predict the values for the rest of the data.

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

1) Decision tree 


```r
predtree <- rpart(classe ~ ., data = trainData, method = "class")
prp(predtree)
```

![](https://github.com/archit2606/Practical-Machine-Learning/blob/master/prediction%20tree.png)<!-- -->
