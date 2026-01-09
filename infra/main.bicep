targetScope = 'subscription'

@description('The name of the resource group to deploy to.')
param resourceGroupName string = 'rg-zz-docs-ai'

@description('The location to deploy the resources to.')
param location string = 'eastus'

@description('The name of the Azure AI Search service.')
param searchServiceName string = 'search-zz-docs-ai-${uniqueString(subscription().id, resourceGroupName)}'

@description('The SKU of the search service.')
@allowed([
  'free'
  'basic'
  'standard'
  'standard2'
  'standard3'
  'storage_optimized_l1'
  'storage_optimized_l2'
])
param sku string = 'basic'

resource rg 'Microsoft.Resources/resourceGroups@2021-04-01' = {
  name: resourceGroupName
  location: location
}

module searchModule 'modules/search/search.bicep' = {
  scope: rg
  name: 'searchModuleDeployment'
  params: {
    location: location
    searchServiceName: searchServiceName
    sku: sku
  }
}

module storageModule 'modules/storage/storage.bicep' = {
  scope: rg
  name: 'storageModuleDeployment'
  params: {
    location: location
    storageAccountName: 'st${uniqueString(subscription().id, resourceGroupName)}'
    containerName: 'documents'
  }
}

output searchServiceName string = searchModule.outputs.searchServiceName
output searchServiceId string = searchModule.outputs.searchServiceId
output storageConnectionString string = storageModule.outputs.storageConnectionString
