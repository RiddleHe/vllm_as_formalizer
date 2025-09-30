(define (problem blocksworld)
    (:domain blocksworld)
    (:objects
        red - block
        green - block
        yellow - block
        purple - block
        orange - block
        blue - block
        robot - robot
    )
    (:init
        (handempty robot)        (ontable red)
        (on green red)
        (clear green)
        (ontable yellow)
        (on purple yellow)
        (on orange purple)
        (on blue orange)
        (clear blue)
        ; red -> green
        ; yellow -> purple -> orange -> blue
    )
    (:goal (and (on green red)
        (on orange purple)
        (on blue yellow)))
)
    