var app = angular.module('botsBoard', []);
app.config(function($interpolateProvider) {
  $interpolateProvider.startSymbol('@@@');
  $interpolateProvider.endSymbol('@@@');
});

var scrum = {
      getStoryDetails: function(stringId) {
        if (stringId== "0, 5, 7") return {"story_points": 2, "story_title": "simple story"};
        if (stringId== "0, 6, 6") return {"story_points": 3, "story_title": "later later later"};
        if (stringId== "4, 2, 5") return {"story_points": 4, "story_title": "unique job"};
        if (stringId== "4, 6, 5") return {"story_points": 5, "story_title": "YAY, so many stories"};
        if (stringId== "5, 0, 1") return {"story_points": 1, "story_title": "yet aANOTHER simple story"};
        if (stringId== "7, 5, 7") return {"story_points": 6, "story_title": "get life"};
        return {"story_points": 7, "story_title": "lorem ipsum easter egg"};
      }
};



function readStories($scope, $http, ocvUrl) {

    var responseString;
    var ocvResult = {};
    var columnsStories = { 1: [ ], 2: [], 3: [] };

    $http.get(ocvUrl).success(function(data){

//        var data = "{\n    \"0, 5, 7\": 3,\n    \"0, 6, 6\": 1,\n    \"4, 2, 5\": 3,\n    \"4, 6, 5\": 2,\n    \"5, 0, 1\": 3,\n    \"7, 5, 7\": 2\n}";    
//ocvResult = JSON.parse(JSON.parse(data));

        ocvResult = JSON.parse(data);
        console.log("ocvResult type arter parse is:"+typeof stringValue);
        
        if (ocvResult) {
            //alert(ocvResult);
            for (var stringId in ocvResult) {
                // console.log(stringId);
                if (columnsStories[ocvResult[stringId]]) {
                    var storyDetails = scrum.getStoryDetails(stringId);
                    //console.log(stringId);
                    columnsStories[ocvResult[stringId]].push({"id":stringId, 
                        "points": storyDetails["story_points"],
                        "title": storyDetails["story_title"]
                    }); 
                }
            }
        } else {
           console.log('ocvResult is null');
        }
        console.log(columnsStories);
        $scope.storiesToDo = columnsStories[1];
        $scope.storiesInProgress = columnsStories[2];
        $scope.storiesDone = columnsStories[3];
    });
}

app.controller('MainCtrl', 
function($scope, $http) {

    var ocvUrl =  '/upload/last.json?delay='+Date.now();

    $scope.detailFrame= ocvUrl;

    readStories($scope, $http, ocvUrl); 

});

