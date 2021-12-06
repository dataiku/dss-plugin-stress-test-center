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
            possibleValues: '=',
            taboo: '=',
            validity: '@',
            onChange: '='
        },
        restrict: 'A',
        templateUrl:'/plugins/stress-test-center/resource/templates/custom-dropdown.html',
        link: function(scope, elem, attrs) {
            const isMulti = !!attrs.items;
            scope.form.$setValidity(scope.validity, false);

            scope.canBeSelected = function(item) {
                if (!scope.taboo) return true;
                return item === scope.item || !(item in scope.taboo);
            }

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
                    if (scope.item === value) return;
                    if (scope.onChange) {
                        scope.onChange(scope.item, value, elem);
                    }
                    scope.item = value;
                }
                scope.form.$setValidity(scope.validity, !!scope.item || !!(scope.items || {}).size);
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
            const labelElem = elem.find(".label-text");
            scope.$on("closeDropdowns", function(e, target) {
                if ((target) && ( angular.element(target).closest(dropdownElem)[0]
                    || angular.element(target).closest(labelElem)[0] )) { return; }
                scope.isOpen = false;
            })
        }
    }
})

// For now, key can only be one of a preset of values (dropdown)
// & the value field only accepts numbers
app.directive("keyValueList", function($timeout) {
    return {
        scope: {
            keyOptions: '=',
            map: '=',
            keyLabel: '@',
            keyItemLabel: '@',
            valueLabel: '@',
            valueRange: '@',
            form: "="
        },
        restrict: 'A',
        templateUrl:'/plugins/stress-test-center/resource/templates/key-value-list.html',
        link: function(scope) {
            scope.keys = [null];
            [ scope.valueMin, scope.valueMax ] = scope.$eval(scope.valueRange) || [null, null];
            const VALIDITY = "key-value-list-valid";
            const DEFAULT_VALUE = .5;
            scope.form.$setValidity(VALIDITY, false);

            scope.deleteListItem = function(index) {
                if (!index) return;
                const removedKey = scope.keys.splice(index, 1)[0];
                delete scope.map[removedKey];
                scope.form.$setValidity(
                    VALIDITY,
                    !!scope.keys.length && scope.keys.every(key => !!key)
                );
            }

            scope.addListItem = function() {
                if (!scope.keys.length) {
                    scope.form.$setValidity(VALIDITY, true);
                }
                scope.keys.push(null);
            }

            scope.dropdownChange = function(oldValue, newValue, keyElem) {
                scope.map[newValue] = scope.map[oldValue] || DEFAULT_VALUE;
                delete scope.map[oldValue];
                $timeout(function() {
                    const valueElem = keyElem.parent().find(".key-value-element__value")[0];
                    valueElem.focus();
                });
            };
        }
    }
});

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
