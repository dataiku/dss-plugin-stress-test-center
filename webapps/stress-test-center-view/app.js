let webAppConfig = dataiku.getWebAppConfig();
let modelId = webAppConfig['modelId'];
let versionId = webAppConfig['versionId'];

(function() {
    'use strict';
    app.controller('vizController', function($scope, $http, $timeout) {
       $scope.activatePS = false;
       $scope.activateAA = false;
       $scope.activateMV = false;
       $scope.activateS = false;

       $scope.highlightPS = true;
       $scope.highlightAA = false;
       $scope.highlightMV = false;
       $scope.highlightS = false;

       var choice_list = {
        'highlight_ps': $scope.highlightPS,
        'highlight_aa': $scope.highlightAA,
        'highlight_mv': $scope.highlightMV,
        'highlight_s': $scope.highlightS
        };

       $scope.activateBoxPS = function(){
            if ($scope.priorShiftActive) {
                $scope.activatePS = false;
            }
            else {
                console.log('Activate PS');
                $scope.activatePS = true;
            }
                $scope.highlightPS = true;
                $scope.highlightAA = false;
                $scope.highlightMV = false;
                $scope.highlightS = false;
       };

       $scope.activateBoxAA = function(){
            if ($scope.aaActive) {
                $scope.activateAA = false;
            }
            else {
                console.log('Activate AA');
                $scope.activateAA = true;
            }
                $scope.highlightAA = true;
                $scope.highlightPS = false;
                $scope.highlightMV = false;
                $scope.highlightS = false;
       };

        $scope.activateBoxMV = function(){
            if ($scope.mvActive) {
                $scope.activateMV = false;
            }
            else {
                console.log('Activate MV');
                $scope.activateMV = true;
            }
                $scope.highlightMV = true;
                $scope.highlightPS = false;
                $scope.highlightAA = false;
                $scope.highlightS = false;
       };

        $scope.activateBoxS = function(){
            if ($scope.sActive) {
                $scope.activateS = false;
            }
            else {
                console.log('Activate S');
                $scope.activateS = true;
            }
            $scope.highlightS = true;
            $scope.highlightPS = false;
            $scope.highlightAA = false;
            $scope.highlightMV = false;
            $scope.highlightS = true;
       };


    })}
)();