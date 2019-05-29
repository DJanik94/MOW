# ----------------- Imports -------------
library(tidyverse)
library(corrplot)
library("data.table")
library(caret)
library(randomForest)
library('e1071')
library(dplyr)
library(Boruta)
library(BBmisc)
library(rlist)
library(pROC)
library(mlr)
library(plotROC)
library(fakeR)

# ---------------- Data read -------------
#setwd('C:/Users/Gabrysia/git_mow')
setwd('~/Studia/SEM2/MOW/Projekt/MOW')


dat <- read.csv("WA_Fn-UseC_-HR-Employee-Attrition.csv", header=TRUE)
dat <-subset(dat, select = -c(Over18, StandardHours, EmployeeCount, EmployeeNumber))

sapply(dat, class)



# -------- Data preparation --------------
incomplete_count <- toString(sum(!complete.cases(dat)))
print(paste("Incomplete cases:", incomplete_count, sep = " "))

#Normalize continous values
dat_normalized <- normalize(dat, method = "range", range = c(0, 1))

####
split_ratio <- c(0.6, 0.2, 0.2)

training_set_size <- round(split_ratio[1]*nrow(dat))
validation_set_size <- round(split_ratio[2]*nrow(dat))
test_set_size <- round(split_ratio[3]*nrow(dat))

tr_indices  <- sample(seq_len(nrow(dat)), size = training_set_size)
training_set <- dat[tr_indices,]
temp_set <- dat[-tr_indices,]
v_indices <- sample(seq_len(nrow(temp_set)), size = validation_set_size)
validation_set  = temp_set[v_indices,]
testing_set = temp_set[-v_indices,]

#--Feature selection using random forest----

# feature_selection_rf <- randomForest(formula = Attrition~., data = training_set)
# print(summary(feature_selection_rf))
# forest_summary <- varImp(feature_selection_rf)
# forest_summary <- data.frame(overall = forest_summary$Overall,
#                             names   = rownames(forest_summary))
# forest_summary_sorted <- forest_summary[order(forest_summary$overall, decreasing = T), ]
# rownames(forest_summary_sorted) <- NULL
# varImpPlot(feature_selection_rf)
# 
# features_sorted <- as.character(forest_summary_sorted$names)
# features_num <- c(10, 15, 20, 26)

#-----Feature selection using Boruta algorithm-----
boruta_summary <- Boruta(Attrition~., data = training_set, doTrace = 2, maxRuns = 500)
print(boruta_summary)
plot(boruta_summary, xlab = "", ylab = "Importance", las = 2)
boruta_attributes <- getSelectedAttributes(boruta_summary)
getConfirmedFormula(boruta_summary)
getNonRejectedFormula(boruta_summary)
attStats(boruta_summary)

boruta_training <- subset(training_set, select = c("Attrition", boruta_attributes))
boruta_validation <- subset(validation_set, select = c("Attrition", boruta_attributes))

#----RANDOM FOREST PREDICTION----------------
trees_num <- c(500, 700, 1500, 3000)
features_num <- ncol(boruta_training)-1
rf_best_accuracy <- c(0.0 )
rf_best_trees_num <- NULL
rf_best_confm <- NULL

  for (t_num in trees_num){
    rf_model <- randomForest(formula = Attrition~., data = boruta_training, ntree = t_num)
    rf_pred <- predict(rf_model, newdata = boruta_validation, type = 'response')
    rf_confm <- caret::confusionMatrix(rf_pred, boruta_validation$Attrition)
    if (rf_confm$overall['Accuracy'] > rf_best_accuracy){
      rf_best_accuracy <- rf_confm$overall['Accuracy']
      rf_best_trees_num <- c(t_num)
      rf_best_confm <- rf_confm
    }
    #cat(sprintf("Confusion matrix for random forest model with %s features\n and %s trees", f_num, t_num))
    #print(rf_confm)
    cat(sprintf("Accuracy for model with %s features and %s trees: %s \n", features_num, t_num,rf_confm$overall['Accuracy']  ))
  }

cat(sprintf("Confusion matrix for the best random forest model with %s features, %s trees
              \n Confusion matrix: \n", features_num, rf_best_trees_num))
print(rf_best_confm)



#----------Numeric data preparation-----------
numeric_tr_subset <- normalize(as.data.table(dataPreparation::shapeSet(boruta_training,
                                  finalForm = 'numerical_matrix',
                                  verbose =TRUE)),
                               method = "range",
                               range = c(0, 1))
numeric_tr_subset <-subset(numeric_tr_subset, select = -c(AttritionNo))
names(numeric_tr_subset) <- str_replace_all(names(numeric_tr_subset), c(" " = "_", "," = "", "&" = ""))
M <- cor(numeric_tr_subset)
head(round(M,2))

numeric_val_subset <- normalize(as.data.table(dataPreparation::shapeSet(boruta_validation,
                                  finalForm = 'numerical_matrix',
                                  verbose =TRUE)),
                                method = "range",
                                range = c(0, 1))
numeric_val_subset <-subset(numeric_val_subset, select = -c(AttritionNo))
names(numeric_val_subset) <- str_replace_all(names(numeric_val_subset), c(" " = "_", "," = "", "&" = ""))
M <- cor(numeric_val_subset)
head(round(M,2))

rf_roc <- roc(AttritionYes ~ rf_pred, data = numeric_val_subset)
plot(rf_roc, xlim=c(1,0), ylim = c(0,1))


# -------------- GLM -----------------
  
glm_model = stats::glm(formula = AttritionYes~., data = numeric_tr_subset)
glm_pred <- predict(glm_model, newdata = numeric_val_subset, method = "glm.fit")
glm_pred <- round(glm_pred)
glm_confm <- caret::confusionMatrix(table(glm_pred, numeric_val_subset$AttritionYes))
print(glm_confm)

glm_roc <- roc(AttritionYes ~ glm_pred, data = numeric_val_subset)
plot(glm_roc, xlim=c(1,0), ylim = c(0,1))

# -------------- ANN -  -----------------
ann_training_desired_output <- subset(numeric_tr_subset, select = c(AttritionYes))
ann_training_input <- subset(numeric_tr_subset, select = -c(AttritionYes))

ann_validation_desired_output <- subset(numeric_val_subset, select = c(AttritionYes))
ann_validation_desired_output$AttritionYes[ann_validation_desired_output == 0] <- "No"
ann_validation_desired_output$AttritionYes[ann_validation_desired_output == 1] <- "Yes"
ann_validation_desired_output$AttritionYes <- as.factor(ann_validation_desired_output$AttritionYes)

ann_validation_input <- subset(numeric_val_subset, select = -c(AttritionYes))     

ann_model_1 <- RSNNS::mlp(x = ann_training_input,
                       y = ann_training_desired_output,
                       size = c(4),
                       maxit = 1000, 
                      initFunc = "Randomize_Weights",
                      initFuncParams = c(-0.3, 0.3),
                      learnFunc = "Std_Backpropagation",
                      linOut = FALSE)

ann_model_2 <- RSNNS::mlp(x = ann_training_input,
                         y = ann_training_desired_output,
                         size = c(4),
                         maxit = 1000, 
                         initFunc = "Randomize_Weights",
                         initFuncParams = c(-0.3, 0.3),
                         learnFunc = "SCG",
                         linOut = FALSE)
ann_predictions <- list()
ann_conf_matrices <- list()
ann_ROCs <- list()
ann_models <- list( ann_model_2)

for (model in ann_models){
  
  ann_pred <- (as.data.frame(round(predict(model, newdata = ann_validation_input))))
  round(as.data.table(predict(model, newdata = ann_validation_input)))
  
  ann_pred$V1[ann_pred$V1 == 0] <- "No"
  ann_pred$V1[ann_pred$V1 == 1] <- "Yes"
  ann_pred$V1 <- as.factor(ann_pred$V1)
  list.append(ann_predictions, ann_pred)
  
  ann_confm <- caret::confusionMatrix(ann_pred$V1,  ann_validation_desired_output$AttritionYes)
  print(ann_confm)
  list.append(ann_conf_matrices,  ann_confm)
  
  ann_roc <- roc(model ~ ann_pred$V1, data = ann_validation_input)
  plot(ann_roc)
  
  
  
}





    
