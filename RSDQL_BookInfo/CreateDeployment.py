from kubernetes import client, config
import yaml

class CreateDeployment:
    def __init__(self):
        # 加载 Kubernetes 配置
        config.load_kube_config()
        # 创建 V1Deployment 对象
        self.api_instance = client.AppsV1Api()


    def createdetails(self,nodeIndex):
        nodeName=self.NodeindexToName(nodeIndex)
        # 读取 YAML 文件
        with open("./yaml/details.yaml", "r") as file:
            deployment_yaml = yaml.safe_load(file)

        # 修改某个配置，例如修改副本数
        deployment_yaml['spec']['template']['spec']['nodeName'] = nodeName

        deployment = self.api_instance.create_namespaced_deployment(
            namespace="bookinfo",
            body=deployment_yaml
        )

        print(f"Deployment created. Name: {deployment.metadata.name}")
        print(f"Deployment created. status='{deployment.status}'")

    def createproductpage(self,nodeIndex):
        nodeName = self.NodeindexToName(nodeIndex)
        # 读取 YAML 文件
        with open("./yaml/productpage.yaml", "r") as file:
            deployment_yaml = yaml.safe_load(file)

        # 修改某个配置，例如修改副本数
        deployment_yaml['spec']['template']['spec']['nodeName'] = nodeName

        deployment = self.api_instance.create_namespaced_deployment(
            namespace="bookinfo",
            body=deployment_yaml
        )

        print(f"Deployment created. Name: {deployment.metadata.name}")
        print(f"Deployment created. status='{deployment.status}'")

    def createrating(self,nodeIndex):
        nodeName = self.NodeindexToName(nodeIndex)
        # 读取 YAML 文件
        with open("./yaml/ratings.yaml", "r") as file:
            deployment_yaml = yaml.safe_load(file)

        # 修改某个配置，例如修改副本数
        deployment_yaml['spec']['template']['spec']['nodeName'] = nodeName

        deployment = self.api_instance.create_namespaced_deployment(
            namespace="bookinfo",
            body=deployment_yaml
        )

        print(f"Deployment created. Name: {deployment.metadata.name}")
        print(f"Deployment created. status='{deployment.status}'")

    def createreviews(self,version,nodeIndex):
        nodeName = self.NodeindexToName(nodeIndex)
        # 读取YAML文件中的所有配置
        with open('./yaml/reviews.yaml', 'r') as file:
            deployment_yaml = list(yaml.safe_load_all(file))


        # 迭代每个Deployment配置
        for deployment in deployment_yaml:
            version -= 1;
            # 修改Deployment配置
            # 假设我们想修改每个Deployment的容器镜像
            if(version==0):
                deployment['spec']['template']['spec']['nodeName'] = nodeName
                # 创建Deployment
                api_response = self.api_instance.create_namespaced_deployment(
                    body=deployment,
                    namespace="bookinfo"  # 指定命名空间，根据需要修改
                )
                print(f"Deployment created. Name: {api_response.metadata.name}")
                print(f"Deployment created. status='{api_response.status}'")
                break

    def NodeindexToName(self,Nodeindex):
        Nodename="node"+str(Nodeindex)
        return Nodename

