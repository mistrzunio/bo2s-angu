var app = angular.module('botsBoard', []);


var scrum = {
      getStoryDetails: function(stringId) {
          return {"story_points": 7, "story_title": "lorem ipsum easter egg"};
      }
};



function readStories($scope, $http, ocvUrl) {

    var responseString;
    var ocvResult = {};
    var columnsStories = { 1: [ ], 2: [], 3: [] };

//    $http.get(ocvUrl).success(function(data){

        var data = "{\n    \"0, 5, 7\": 3,\n    \"0, 6, 6\": 1,\n    \"4, 2, 5\": 3,\n    \"4, 6, 5\": 2,\n    \"5, 0, 1\": 3,\n    \"7, 5, 7\": 2\n}";
        ocvResult = JSON.parse(data)

        if (ocvResult) {
            for (var stringId in ocvResult) {
                if (columnsStories[ocvResult[stringId]]) {
                    var storyDetails = scrum.getStoryDetails(stringId);
                    columnsStories[ocvResult[stringId]].push({"id":stringId, 
                        "points": storyDetails["story_points"],
                        "title": storyDetails["story_title"]
                    }); 
                }
            }
        } else {
           console.log('ocvResult is null');
        }

        $scope.storiesToDo = columnsStories[1];
        $scope.storiesInProgress = columnsStories[2];
        $scope.storiesDone = columnsStories[3];
//    });
}

app.controller('MainCtrl', 
function($scope, $http) {

    var ocvUrl =  'https://bo2s-tumefacient-godliness.run.aws-usw02-pr.ice.predix.io/upload/last.json';

    $scope.detailFrame= ocvUrl;

    readStories($scope, $http, ocvUrl); 

});