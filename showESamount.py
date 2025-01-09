import matplotlib.pyplot as plt

# 假设有以下数据
concurrent_requests = [0,100, 200, 300, 400]
Productpage = [1,2,4, 5, 6]
Detail = [1,1 ,2, 4, 5]
Rating = [1,1 ,3, 4, 4]
Reviews = [3, 5,6, 6, 7]

plt.plot(concurrent_requests, Productpage, 'ks-', label='Productpage')
plt.plot(concurrent_requests, Detail, 'ro-', label='Detail')
plt.plot(concurrent_requests, Rating, 'b^-', label='Rating')
plt.plot(concurrent_requests, Reviews, 'y*-', label='Reviews')

plt.xlabel('Number of concurrent requests')
plt.ylabel('Container Number')
plt.legend()
plt.show()