# Plantilla para el Procesado de datos
####################################################################

# Importar el data set
dataset=read.csv("Data.csv")
# Aqui no hay que hacer una division entre la variable que quiero predecir

####################################################################

# Tratamiendo de los valores NA

# ifelse(condicion,siesverdad,siesfalso) 
dataset$Age=ifelse(is.na(dataset$Age),ave(dataset$Age,FUN = function(x) mean(x,na.rm = TRUE)) ,dataset$Age)
dataset$Salary=ifelse(is.na(dataset$Salary), ave(dataset$Salary,FUN = function(x) mean(x,na.rm = TRUE)) ,dataset$Salary)

####################################################################

# Codificar las variables categoricas
dataset$Country=factor(dataset$Country,levels=c("France","Spain","Germany"),labels=c(1,2,3))
dataset$Purchased=factor(dataset$Purchased,levels = c("No","Yes"),labels = c(0,1))

####################################################################

# Split entre Entrenamiento y testing data  
# install.packages("caTools")
# library(caTools)
# Inicia el generador de numeros aleatorios en 123
set.seed(123)
# Dividir el 80% de la variable independiente para data de entrenamiento
split=sample.split(dataset$Purchased,SplitRatio = 0.8)
# Entrenamiento tendran true
training_set=subset(dataset,split==TRUE)
# Testing tendran false
testing_set=subset(dataset,split==FALSE)

####################################################################

# Escalar los datos(normalizacion)
# todas las filas pero columnas dos y tres
training_set[,2:3]=scale(training_set[,2:3])
testing_set[,2:3]=scale(testing_set[,2:3])
