Usually, after implementing an application in a development environment, we need to deploy it to a production environment on the customer's side or to a cloud service platform. 
Services have become very popular because you can configure the computational environments for your customer's needs with an excellent balance ratio between cost, scalability, and performance.
Also, the use of such services eliminates the need for your customers to maintain the hardware devices they're using. 

Lets learn how to deploy a simple image classification applicatoiion to the Google Compute Engine Platform. 
Initially, we need to develop a test such an application in a development environment. 
We are going to make an HTTP service that responds to "POST" requests with image data encoded in multipart format. 
Let's start by implementing the server. 
