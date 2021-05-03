# What is data normalisation?

("The process of efficiently _organising_ data:")
1. eliminating duplicates
2. logically structuring "data dependencies"  
   -> sorting by digit? probably not necessary here


In image analysis:
- standardisation aka mean = 0 and sd = 1  
  -> either calculated per image or per entire dataset
- stretching aka clipping intensities/ noise reduction  
  -> images already very nice: most pixels that dont contribute to digit are 0
- (scaling: bring images of different formats to same range = "normalisation")

=> test the effect of each to see whats best   

Other things to maybe do:
- any NA? are all values between 0 and 255?
- Images the right way up? -> how on earth do i check that ?!1?
- are there duplicates? and does removing them interfere with the program?
- are numbers correctly assigned?