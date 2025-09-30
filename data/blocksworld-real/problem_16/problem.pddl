(define (problem blocksworld)
    (:domain blocksworld)
    (:objects
        green - block
        yellow - block
        red - block
        blue - block
        orange - block
        purple - block
        robot - robot
    )
    (:init
        (handempty robot)        (ontable green)
        (clear green)
        (ontable yellow)
        (on red yellow)
        (clear red)
        (ontable blue)
        (on orange blue)
        (on purple orange)
        (clear purple)
        ; green
        ; yellow -> red
        ; blue -> orange -> purple
    )
    (:goal (and (on blue yellow)
        (on purple blue)
        (on orange purple)
        (on red orange)
        (on green red)))
)
    