# Demonstrate R's sample(g, k) gotcha inside .balance_indices when a class has one member in the fitting split
train_y <- c(1,1,1,2,2,2,3)   # class 3 has a single row (index 7)
groups <- split(seq_along(train_y), train_y)
set.seed(1); down <- unlist(lapply(groups, function(g) sample(g, 1, replace=FALSE)))
set.seed(1); up <- unlist(lapply(groups, function(g) sample(g, 3, replace=TRUE)))
cat("groups$`3` =", groups$`3`, "\n down picks for class 3:", down[3], "(class", train_y[down[3]], ")\n up picks for class 3:", tail(up,3), "(classes", train_y[tail(up,3)], ")\n")
