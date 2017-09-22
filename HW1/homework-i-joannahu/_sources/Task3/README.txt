===========================================
README
===========================================


Welcome to Applied Machine Learning HW1 Task 3.1 Function documentation. 



Function (mean,std) meanstd(X,axis) computes mean and standard deviation of input data.



.. math::

   

  mean = \frac{\sum x_i }{N}

   
  
  std = \sqrt{\frac{(\sum x_i - mean)^2}{N}}
    
  

  Let  X =
  \begin{array}{ccc}
  1 & 2 & 3 \\
  4 & 5 & 6 \\
  7 & 8 & 9
  \end{array}


  
  if  axis = 0, mean = 
  \begin{array}{c}

  3 \\
   
  7.5 \\
   
  12    
  \end{array}

  if axis = 1, mean = 
  \begin{array}{ccc}

  6 & 7.5 & 9  
  \end{array}

  if  axis = NULL, mean = 45
 