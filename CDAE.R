library(keras)

create_cdae <- function(I, U, K=NULL, q=NULL, l=NULL, hidden_activation='relu', output_activation='sigmoid'){

  # Use default parameter values if they were not provided in the function call
  if(is.null(K)){
    K = 50 # Dimension of hidden layer
  }  
  if(is.null(q)){
    q = .2 # Drop probability
  }  
  if(is.null(K)){
    l = .1 # Regularization parameter
  }  

  item_input <- layer_input(shape = I, name = "item_input") 
  user_input <- layer_input(shape = 1, name = "user_input")
  
  item_hidden <- 
    item_input %>%
    layer_dropout(rate = q, name = "dropout") %>% #TODO: Scale remaining items? See paper.
    layer_dense(units = K, 
                kernel_regularizer = regularizer_l2(l),
                bias_regularizer = regularizer_l2(l),
                name = "item_hidden") 
  
  user_hidden <-
    user_input %>%
    layer_embedding(input_dim = U, 
                    output_dim = K, 
                    input_length = 1,
                    embeddings_regularizer = regularizer_l2(l),
                    name = "user_embedding") %>% 
    layer_flatten()
  
  hidden_layer <- 
    layer_add(inputs = list(item_hidden, user_hidden)) %>%
    layer_activation(hidden_activation, name = "hidden_layer") 
  
  output <- 
    hidden_layer %>%
    layer_dense(units = I, 
                        kernel_regularizer = regularizer_l2(l),
                        bias_regularizer = regularizer_l2(l),
                        activation = output_activation,
                        name = "output") 
  
  model <- keras_model(list(item_input, user_input), output)
  
  # Compile model 
  model %>% compile(
    optimizer = "adagrad",
    loss = "binary_crossentropy",   
    metrics = c("accuracy")
  )
  
  #TODO: Use negative sampling to avoid updating all weights.    
}


