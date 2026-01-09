@description('The location to deploy the resources to.')
param location string

@description('The name of the Azure AI Search service.')
param searchServiceName string

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

module searchService 'br/public:avm/res/search/search-service:0.12.0' = {
  name: 'searchServiceDeployment'
  params: {
    name: searchServiceName
    location: location
    sku: sku
    semanticSearch: 'free'
    authOptions: {
      aadOrApiKey: {
        aadAuthFailureMode: 'http401WithBearerChallenge'
      }
    }
    disableLocalAuth: false
    publicNetworkAccess: 'Enabled'
  }
}

output searchServiceName string = searchService.outputs.name
output searchServiceId string = searchService.outputs.resourceId
