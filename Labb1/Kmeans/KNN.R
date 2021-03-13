#Load
Iris <- read.table("iris.data", sep=",")
names(Iris)[names(Iris) == "V1"] <- "Sepal.Length"
names(Iris)[names(Iris) == "V2"] <- "Sepal.Width"
names(Iris)[names(Iris) == "V3"] <- "Petal.Length"
names(Iris)[names(Iris) == "V4"] <- "Petal.Width"
names(Iris)[names(Iris) == "V5"] <- "Class"

#Setting seed an shuffels the dataset
gp <-runif(nrow(Iris))
Iris2 <- Iris[order(gp),]

#Check the ranges of the different attributes
summary(Iris2[,c(1,2,3,4)])

#Function to normalize the values
normalize <- function(x) {
  return( (x-min(x)) / (max(x) - min(x)))
}

#Normalize the dataset
Iris_n <- as.data.frame(lapply(Iris2[,c(1,2,3,4)], normalize))

#Create the different datasets
training <- Iris_n[1:109, ]
testing <- Iris_n[110:150, ]

training_target <- Iris2[1:109, 5]
testing_target <- Iris2[110:150, 5]

#KNN is located in class
require(class)


#Run KNN with the testing and training dataset
m1 <- knn(train = training, test = testing, cl = training_target, k = 13)
#Confusion Matrix
tb <- table(testing_target, m1)

tb <- as.matrix(tb)
tb

accuracy <- sum(diag(tb))/41