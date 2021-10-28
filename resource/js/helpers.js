'use strict';

app.service("ModalService", function() {
    const remove = function(config) {
        return function(event) {
            if (event && !event.target.className.includes("dku-modal-background")) return false;
            for (const key in config) {
                delete config[key];
            }
            return true;
        }
    };
    return {
        create: function(config) {
            return {
                confirm: function(msg, title, confirmAction) {
                    Object.assign(config, {
                        type: "confirm",
                        msg: msg,
                        title: title,
                        confirmAction: confirmAction
                    });
                },
                error: function(msg) {
                    Object.assign(config, {
                        type: "error",
                        msg: msg,
                        title: "Backend error"
                    });
                },
                alert: function(msg, title) {
                    Object.assign(config, {
                        type: "alert",
                        msg: msg,
                        title: title
                    });
                },
                prompt: function(inputLabel, confirmAction, res, title, msg, attrs) {
                    Object.assign(config, {
                        type: "prompt",
                        inputLabel: inputLabel,
                        promptResult: res,
                        title: title,
                        msg: msg,
                        conditions: attrs,
                        confirmAction: function() {
                            confirmAction(config.promptResult);
                        }
                    });
                }
            };
        },
        remove: remove
    }
});

app.directive("modalBackground", function($compile) {
    return {
        scope: true,
        restrict: "C",
        templateUrl: "/plugins/stress-test-center/resource/templates/modal.html",
        link: function(scope, element) {
            if (scope.modal.conditions) {
                const inputField = element.find("input");
                for (const attr in scope.modal.conditions) {
                    inputField.attr(attr, scope.modal.conditions[attr]);
                }
                $compile(inputField)(scope);
            }
        }
    }
});

app.directive("spinner", function () {
    return {
        template: "<div class='spinner-container'></div>",
        link: function (scope, element) {
            var opts = {
                lines: 6,
                length: 0,
                width: 10,
                radius: 10,
                corners: 1,
                rotate: 0,
                color: '#fff',
                speed: 2,
                trail: 60,
                shadow: false,
                hwaccel: false,
                className: 'spinner',
                zIndex: 2e9,
                top: '10px',
                left: '10px'
             };
             const spinner = new Spinner(opts);
             spinner.spin(element[0].childNodes[0]);
        }
    }
});

app.directive("customDropdown", function() {
    return {
        scope: {
            form: '=',
            itemImage: '=?',
            label: '@',
            itemName: '@',
            item: '=',
            items: '=',
            possibleValues: '='
        },
        restrict: 'A',
        templateUrl:'/plugins/stress-test-center/resource/templates/custom-dropdown.html',
        link: function(scope, elem, attrs) {
            scope.form.$setValidity("dropdown-not-empty", false);

            const isMulti = attrs.items;

            scope.isSelected = function(value) {
                if (isMulti) {
                    return scope.items.has(value);
                }
                return scope.item === value;
            }

            scope.updateSelection = function(value, event) {
                if (isMulti) {
                    if (scope.isSelected(value)) {
                        scope.items.delete(value);
                    } else {
                        scope.items.add(value);
                    }
                    event.stopPropagation();
                } else {
                    scope.item = value;
                }
                scope.form.$setValidity("dropdown-not-empty", !!scope.item || !!(scope.items || {}).size);
            }

            scope.getPlaceholder = function() {
                if (isMulti) {
                    if (!(scope.items || {}).size) return "Select " + scope.itemName + "s";
                    return scope.items.size + " " + scope.itemName + (scope.items.size > 1 ? "s" : "");
                }
                if (!scope.item) return "Select a " + scope.itemName;
                return scope.item;
            }

            scope.toggleDropdown = function() {
                scope.isOpen = !scope.isOpen;
            }

            const dropdownElem = elem.find(".custom-dropdown");
            scope.$on("closeDropdowns", function(e, target) {
                if (target && angular.element(target).closest(dropdownElem)[0]) return;
                scope.isOpen = false;
            })
        }
    }
})

app.directive("helpIcon", function () {
    return {
        restrict: 'E',
        scope: {
            helpText: '@',

        },
        template: `<i class="icon-info-sign icon--hoverable" ng-mouseover="showTooltip()" ng-mouseleave="showTooltip()">
            <div class="help-text__tooltip tooltip--hidden">
                <i class="icon-info-sign"></i>
                {{ helpText }}
            </div>
        </i>`,
        link: function(scope, elem) {
            scope.showTooltip = function() {
                const top = elem[0].getBoundingClientRect().top;
                const tooltip = elem.find(".help-text__tooltip");
                tooltip.css("top", (top - 8) + "px");
                tooltip.toggleClass("tooltip--hidden");
            }
        }
    }
});

app.filter("toFixedIfNeeded", function() {
    return function(number, decimals) {
        if(Math.round(number) !== number) {
            return parseFloat(number.toFixed(decimals));
        }
        return number;
    }
});
