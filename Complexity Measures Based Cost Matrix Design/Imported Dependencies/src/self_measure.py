class IllegalArgumentError(ValueError): 
    def __init__(self, illegal_argument, options):
        self.illegal_argument = illegal_argument
        self.message = f"{illegal_argument} not within these options: {', '.join(options)}" 
        super().__init__(self.message)
        
def cal_class_weight(measure,value):
    C_W = {}
    if measure=="IR":
        C_W[0] = round((1/value)*100,2)
        C_W[1] = 100
    else:
        C_W[0] = round(value*100,2)
        C_W[1] = 100
    return C_W