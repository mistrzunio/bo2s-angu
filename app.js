var app = angular.module('botsBoard', []);
var lastJsonResponse;  



function readStories($scope, $http) {

    var ocvUrl =  'https://bo2s-tumefacient-godliness.run.aws-usw02-pr.ice.predix.io/upload/last.json';
    var responseString;
    var ocvResult = {};
    var columnsStories = { 1: [ ], 2: [], 3: [] };

    $http.get(ocvUrl).success(function(data){

        ocvResult = JSON.parse(data)

        if (ocvResult) {
            for (var stringId in ocvResult) {
                if (columnsStories[ocvResult[stringId]]) { 
                  columnsStories[ocvResult[stringId]].push({"id":stringId}); 
                }
            }
        } else {
           console.log('ocvResult is null');
        }

        $scope.storiesToDo = columnsStories[1];
        $scope.storiesInProgress = columnsStories[2];
        $scope.storiesDone = columnsStories[3];
    });
}

app.controller('MainCtrl', 
function($scope, $http) {
    readStories($scope, $http); 

});