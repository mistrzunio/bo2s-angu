var app = angular.module('flapperNews', []);

function readStories() {

  var ocvUrl =  'https://bo2s-tumefacient-godliness.run.aws-usw02-pr.ice.predix.io/upload/last.json';
  var ocvResult = {"0, 5, 7": 3, "0, 6, 6": 1, "4, 2, 5": 3, "4, 6, 5": 2, "5, 0, 1": 3, "7, 5, 7": 2};  
  var columnsStories = { 1: [ ], 2: [], 3: [] };


  
  for (stringId in ocvResult) {
    if (columnsStories[ocvResult[stringId]]) { 
      columnsStories[ocvResult[stringId]].push({"id":stringId}); 
    }
  }
  
  return columnsStories;
}

app.controller('MainCtrl', [
'$scope',
function($scope){
/*
  var ocvUrl =  'https://bo2s-tumefacient-godliness.run.aws-usw02-pr.ice.predix.io/upload/last.json';
  $http({
        method : "GET",
        url : ocvUrl
    }).then(function mySucces(response) {
        alert('response.data');
    }, function myError(response) {
        alert('error!');
    });
*/     
  var storiesStruct = readStories(); 
  $scope.storiesToDo = storiesStruct[1];
  $scope.storiesInProgress = storiesStruct[2];
  $scope.storiesDone = storiesStruct[3];
    
  
}]);