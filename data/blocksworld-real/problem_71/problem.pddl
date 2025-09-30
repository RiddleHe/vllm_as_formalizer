(define (problem blocksworld)
    (:domain blocksworld)
    (:objects
        yellow - block
        blue - block
        orange - block
        red - block
        green - block
        purple - block
        robot - robot
    )
    (:init
        (handempty robot)        (ontable yellow)
        (on blue yellow)
        (clear blue)
        (ontable orange)
        (on red orange)
        (clear red)
        (ontable green)
        (clear green)
        (ontable purple)
        (clear purple)
        ; yellow -> blue
        ; orange -> red
        ; green
        ; purple
    )
    (:goal (and (on purple blue)
        (on red orange)
        (on yellow green)))
)
    