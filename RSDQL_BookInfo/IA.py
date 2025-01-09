from CreateDeployment import  CreateDeployment

#IA基线算法
if __name__ == '__main__':
    createdeployment = CreateDeployment()
    createdeployment.createproductpage(1)
    createdeployment.createdetails(1)
    createdeployment.createrating(1)
    createdeployment.createreviews(1, 1)
    createdeployment.createreviews(2, 1)
    createdeployment.createreviews(3, 1)