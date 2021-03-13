#Load data and fix collum names
Iris <- read.table("iris.data", sep=",")
names(Iris)[names(Iris) == "V1"] <- "Sepal.Length"
names(Iris)[names(Iris) == "V2"] <- "Sepal.Width"
names(Iris)[names(Iris) == "V3"] <- "Petal.Length"
names(Iris)[names(Iris) == "V4"] <- "Petal.Width"
names(Iris)[names(Iris) == "V5"] <- "Class"


#Remove labels
Iris.features = Iris
Iris.features$Class <- NULL

#Run Kmeans
resutlts <- kmeans(Iris.features, 3)

#Table and Plots
#Confusion Matrix
tb <-table(Iris$Class, resutlts$cluster)

#Plotting Kmeans result
plot(Iris[c("Petal.Length", "Petal.Width")], col = resutlts$cluster)
plot(Iris[c("Sepal.Length", "Sepal.Width")], col = resutlts$cluster)

#Plotting Actual Clusters
plot(Iris[c("Petal.Length", "Petal.Width")], col = Iris$Class)
plot(Iris[c("Sepal.Length", "Sepal.Width")], col = Iris$Class)

#Showing table
tb
#Summing diagonal (Only works when the table is built with a diagonal)
diagonal <- sum(diag(tb[nrow(tb):1, ]))
accuracy <- diagonal/nrow(Iris)