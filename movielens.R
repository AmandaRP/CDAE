# This script trains the cdae model on movielens data. 
# Use to recommend new movies to users.

# Load libraries -----------------------------------------------------------

library(tidyverse)
library(magrittr)

# Read and wrangle data ---------------------------------------------------
# Data available from https://grouplens.org/datasets/movielens/

# Small dataset:
filename <- "ml-latest-small.zip"
download.file("http://files.grouplens.org/datasets/movielens/ml-latest-small.zip", filename)
ratings <- read_csv("ml-latest-small/ratings.csv")

# ml-10m dataset:
#filename <- "ml-10m.zip"
#download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", filename)
#ratings <- 
#  read_delim("ml-10M100K/ratings.dat", delim = ":", col_names = FALSE) %>%
#  rename(userId = X1, movieId = X3, rating = X7) %>%
#  select(!starts_with("X")) 
  
# ml-25m dataset
#filename <- "ml-25m.zip"
#download.file("http://files.grouplens.org/datasets/movielens/ml-25m.zip", filename)
#ratings <- read_csv("ml-25m/ratings.csv")

#Keep only ratings greater than or equal to 4 (for translation to binary data).
ratings %<>% filter(rating >=4 )

# Iteratively remove users and items that have fewer than 5 ratings.
threshold <- 5
stop <- FALSE
while(!stop){
  num_ratings_before <- nrow(ratings)
  ratings %<>% inner_join(ratings %>% group_by(userId) %>% count() %>% filter(n >= threshold) %>% select(userId))
  ratings %<>% inner_join(ratings %>% group_by(movieId) %>% count() %>% filter(n >= threshold) %>% select(movieId))
  stop <- if_else(num_ratings_before == nrow(ratings), TRUE, FALSE)
}
  
#Unique list of users and items
userIDs <- ratings %>% select(userId) %>% distinct()
movieIDs <- ratings %>% select(movieId) %>% distinct()

# Create training and test sets -------------------------------------------

#For each user, hold out 20% of ratings for test:
ratings_test <-
  ratings %>% 
  group_by(userId) %>% 
  slice_sample(prop = 0.2) %>% 
  ungroup()

#Define training set by removing test set:
ratings_train <- ratings %>% anti_join(ratings_test, by = c("userId", "movieId"))

#Create binary rating matrices:
ratings_train_matrix <- 
  ratings_train %>% 
  mutate(rating = 1) %>%
  select(userId, movieId, rating) %>% 
  bind_rows(movieIDs %>% bind_cols(userId = max(userIDs) + 1, rating = 0)) %>% #Add a fake user that has entire set of items (so that train and test will have same items).
  pivot_wider(names_from = movieId, names_sort = TRUE, values_from = rating, values_fill = 0) %>%
  filter(userId != max(userIDs) + 1) %>% #remove fake user
  select(-userId) 

ratings_test_matrix <- 
  ratings_test %>% 
  mutate(rating = 1) %>%
  select(userId, movieId, rating) %>% 
  bind_rows(movieIDs %>% bind_cols(userId = max(userIDs) + 1, rating = 0)) %>% #Add a fake user that has entire set of items (so that train and test will have same items).
  pivot_wider(names_from = movieId, names_sort = TRUE, values_from = rating, values_fill = 0) %>%
  filter(userId != max(userIDs) + 1) %>% #remove fake user
  select(-userId) 

#Check that the dimensions of the test and train matrices are the same.
if(any(dim(ratings_train_matrix) != dim(ratings_test_matrix))){
  stop("Train and test ratings matrices do not have the same dimension.")
}


# Create model ------------------------------------------------------------

#TODO: Implement 5-fold CV to pick best hyperparams (including number of epochs)

source("CDAE.R")
model <- create_cdae(I = nrow(movieIDs), 
                     U = max(userIDs), 
                     K = 50, 
                     q = 0.2, 
                     l = 0.1, 
                     hidden_activation='sigmoid') #TODO: Paper uses sigmoid. I use relu as default. Test which is better.
summary(model)

# Train model -------------------------------------------------------------

history <- 
  model %>% 
  fit(
    x = list(item_input = as.matrix(ratings_train_matrix),
             user_input = as.array(userIDs$userId)), 
      y = as.matrix(ratings_train_matrix),
    epochs = 3,
    batch_size = 128, 
    shuffle = TRUE
  ) 


# Evaluate results --------------------------------------------------------

history
#plot(history)

(results <- model %>% evaluate(list(as.matrix(ratings_test_matrix), as.array(userIDs$userId)), as.matrix(ratings_test_matrix))) 

# Get predictions for test set:
test_pred <- 
  model %>% 
  predict(x = list(as.matrix(ratings_test_matrix), as.array(userIDs$userId)))
