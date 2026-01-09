@description('The location to deploy the resources to.')
param location string

@description('The name of the Storage Account.')
param storageAccountName string

@description('The name of the Blob Container to create.')
param containerName string = 'documents'

module storageAccount 'br/public:avm/res/storage/storage-account:0.31.0' = {
  name: 'storageAccountDeployment'
  params: {
    name: storageAccountName
    location: location
    skuName: 'Standard_LRS'
    kind: 'StorageV2'
    minimumTlsVersion: 'TLS1_2'
    supportsHttpsTrafficOnly: true
    allowBlobPublicAccess: false
    blobServices: {
      containers: [
        {
          name: containerName
          publicAccess: 'None'
        }
      ]
    }
    networkAcls: {
        defaultAction: 'Allow'
        bypass: 'AzureServices'
    }
  }
}

// Reference the created resource to get keys reliably regardless of module output names
resource st 'Microsoft.Storage/storageAccounts@2023-01-01' existing = {
  name: storageAccountName
  dependsOn: [
    storageAccount
  ]
}

output storageAccountName string = st.name
output storageAccountId string = st.id
output storageConnectionString string = 'DefaultEndpointsProtocol=https;AccountName=${st.name};AccountKey=${listKeys(st.id, '2023-01-01').keys[0].value};EndpointSuffix=${environment().suffixes.storage}'
