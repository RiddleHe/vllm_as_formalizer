(define (problem blocksworld)
    (:domain blocksworld)
    (:objects
        yellow - block
        orange - block
        purple - block
        red - block
        green - block
        blue - block
        robot - robot
    )
    (:init
        (handempty robot)        (ontable yellow)
        (clear yellow)
        (ontable orange)
        (on purple orange)
        (clear purple)
        (ontable red)
        (clear red)
        (ontable green)
        (on blue green)
        (clear blue)
        ; yellow
        ; orange -> purple
        ; red
        ; green -> blue
    )
    (:goal (and (on yellow blue)
        (on orange purple)
        (on red green)))
)
    